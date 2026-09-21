from __future__ import annotations

import asyncio
import os
import runpy
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock
from urllib.parse import unquote, urlsplit

import pytest
import yaml

pytest.importorskip("dapr")

from examples.memory import dapr_session_example as example


@pytest.fixture
def docker(monkeypatch):
    monkeypatch.setenv("POSTGRES_PASSWORD", "synthetic-unique-password")
    monkeypatch.setattr(example.shutil, "which", lambda name: "/synthetic/docker")
    run = Mock(return_value=subprocess.CompletedProcess([], 0, stdout="", stderr=""))
    monkeypatch.setattr(example.subprocess, "run", run)
    return run


def test_setup_uses_loopback_and_matching_private_credentials(
    tmp_path, docker, monkeypatch, capsys
):
    password = "synthetic 'quoted' \\ value\" # : / @ ? % 密碼 🔑 \u0085"
    monkeypatch.setenv("POSTGRES_PASSWORD", password)
    components = tmp_path / "components"

    example.setup_environment(str(components))

    assert docker.call_args_list[0].args[0] == [
        "docker",
        "container",
        "ls",
        "--all",
        "--format",
        "{{.Names}}",
    ]
    redis, postgres = docker.call_args_list[1:]
    assert redis.args[0] == [
        "docker",
        "run",
        "-d",
        "--name",
        "dapr_redis",
        "-p",
        "127.0.0.1:6379:6379",
        "redis:7-alpine",
    ]
    assert postgres.args[0] == [
        "docker",
        "run",
        "-d",
        "--name",
        "dapr_postgres",
        "-p",
        "127.0.0.1:5432:5432",
        "-e",
        "POSTGRES_USER=postgres",
        "-e",
        "POSTGRES_PASSWORD",
        "-e",
        "POSTGRES_DB=dapr",
        "postgres:16-alpine",
    ]
    assert postgres.kwargs["env"]["POSTGRES_PASSWORD"] == password
    for call in docker.call_args_list:
        assert password not in repr(call.args)
        assert call.kwargs["capture_output"] is True

    pg = yaml.safe_load((components / "statestore-postgres.yaml").read_text())
    metadata = pg["spec"]["metadata"]
    assert len(metadata) == 1
    assert metadata[0]["name"] == "connectionString"
    connection = urlsplit(metadata[0]["value"])
    assert connection.scheme == "postgresql"
    assert connection.username == "postgres"
    assert unquote(connection.password) == password
    assert connection.hostname == "127.0.0.1"
    assert connection.port == 5432
    assert connection.path == "/dapr"
    assert connection.query == connection.fragment == ""
    for filename, name in [
        ("statestore-redis.yaml", "statestore-redis"),
        ("statestore.yaml", "statestore"),
    ]:
        component = yaml.safe_load((components / filename).read_text())
        assert component["metadata"]["name"] == name
        assert component["spec"]["metadata"][0]["value"] == "127.0.0.1:6379"
    if os.name != "nt":
        assert all(path.stat().st_mode & 0o777 == 0o600 for path in components.iterdir())
    output = capsys.readouterr()
    assert password not in output.out + output.err
    assert "Environment setup complete" in output.out


@pytest.mark.parametrize(
    "password", [None, "", "  ", "postgres", " POSTGRES ", "postgres\nunique-suffix", "synthetic\r"]
)
def test_invalid_password_has_no_side_effects(tmp_path, docker, monkeypatch, password):
    if password is None:
        monkeypatch.delenv("POSTGRES_PASSWORD", raising=False)
    else:
        monkeypatch.setenv("POSTGRES_PASSWORD", password)
    components = tmp_path / "components"
    with pytest.raises(SystemExit, match="Set POSTGRES_PASSWORD"):
        example.setup_environment(str(components))
    docker.assert_not_called()
    assert not components.exists()


@pytest.mark.parametrize("name", ["dapr_redis", "dapr_postgres"])
@pytest.mark.parametrize("overwrite", [False, True])
def test_existing_container_is_never_started_or_changed(tmp_path, docker, name, overwrite):
    docker.return_value.stdout = f"unrelated\n{name}\n"
    components = tmp_path / "components"
    with pytest.raises(SystemExit, match="migrate manually"):
        example.setup_environment(str(components), overwrite=overwrite)
    assert docker.call_count == 1
    assert not components.exists()


def test_mismatched_component_stops_before_any_resources_change(tmp_path, docker):
    old = tmp_path / "statestore-postgres.yaml"
    old.write_text("operator-managed component")
    with pytest.raises(SystemExit, match="--overwrite"):
        example.setup_environment(str(tmp_path))
    docker.assert_not_called()
    assert list(tmp_path.iterdir()) == [old]
    assert old.read_text() == "operator-managed component"


def test_overwrite_is_explicit_and_preserves_unrelated_files(tmp_path, docker):
    old = tmp_path / "statestore-postgres.yaml"
    old.write_text("operator-managed component")
    other = tmp_path / "other.yaml"
    other.write_text("unrelated")
    example.setup_environment(str(tmp_path), overwrite=True)
    assert "synthetic-unique-password" in old.read_text()
    assert other.read_text() == "unrelated"
    if os.name != "nt":
        assert old.stat().st_mode & 0o777 == 0o600


def test_matching_components_can_be_reused_without_overwrite(tmp_path, docker):
    example.setup_environment(str(tmp_path))
    original = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    docker.reset_mock()
    example.setup_environment(str(tmp_path))
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == original
    assert docker.call_count == 3


def test_component_symlink_is_not_followed_even_with_overwrite(tmp_path, docker):
    outside = tmp_path / "outside"
    outside.write_text("unrelated")
    components = tmp_path / "components"
    components.mkdir()
    (components / "statestore-postgres.yaml").symlink_to(outside)
    with pytest.raises(SystemExit, match="regular file"):
        example.setup_environment(str(components), overwrite=True)
    docker.assert_not_called()
    assert outside.read_text() == "unrelated"
    assert len(list(components.iterdir())) == 1


@pytest.mark.parametrize("failure", ["missing", "unavailable"])
def test_docker_preflight_failure_does_not_write_files(tmp_path, docker, monkeypatch, failure):
    if failure == "missing":
        monkeypatch.setattr(example.shutil, "which", lambda name: None)
    else:
        docker.return_value.returncode = 1
    with pytest.raises(SystemExit, match="Docker"):
        example.setup_environment(str(tmp_path / "components"))
    assert not list(tmp_path.iterdir())
    assert docker.call_count == (0 if failure == "missing" else 1)


def test_docker_creation_failure_is_reported_without_secret_output(tmp_path, docker, capsys):
    docker.side_effect = [
        subprocess.CompletedProcess([], 0, stdout="", stderr=""),
        subprocess.CompletedProcess(
            [], 1, stdout="synthetic-unique-password", stderr="synthetic-unique-password"
        ),
    ]
    with pytest.raises(SystemExit, match="partially created") as error:
        example.setup_environment(str(tmp_path))
    assert docker.call_count == 2
    assert "synthetic-unique-password" not in str(error.value)
    captured = capsys.readouterr()
    assert "synthetic-unique-password" not in captured.out + captured.err
    assert "Environment setup complete" not in captured.out


def test_instructions_use_the_same_setup_helper_and_never_print_password(docker, capsys):
    asyncio.run(example.setup_instructions())
    output = capsys.readouterr().out
    assert "POSTGRES_PASSWORD" in output
    assert "--setup-env --only-setup" in output
    assert "127.0.0.1" in output
    assert "synthetic-unique-password" not in output
    assert "docker run" not in output
    docker.assert_not_called()


def test_cli_setup_only_does_not_run_demos(tmp_path, docker, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dapr_session_example.py",
            "--setup-env",
            "--only-setup",
            "--components-dir",
            str(tmp_path),
        ],
    )
    monkeypatch.setattr(asyncio, "run", Mock(side_effect=AssertionError("Must not run demos")))
    with pytest.raises(SystemExit) as error:
        runpy.run_path(str(Path(example.__file__)), run_name="__main__")
    assert error.value.code == 0
    assert docker.call_count == 3


def test_cli_help_describes_required_credential_without_effects(docker, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["dapr_session_example.py", "--help"])
    with pytest.raises(SystemExit) as error:
        runpy.run_path(str(Path(example.__file__)), run_name="__main__")
    assert error.value.code == 0
    output = capsys.readouterr().out
    assert "POSTGRES_PASSWORD" in output
    assert "loopback-only" in output
    assert "existing named containers" in " ".join(output.split())
    docker.assert_not_called()
