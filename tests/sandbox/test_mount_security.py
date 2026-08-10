from __future__ import annotations

import asyncio
import builtins
import importlib
import json
import logging
from collections.abc import Iterator, Mapping
from pathlib import Path, PureWindowsPath
from typing import Any, ClassVar, Literal, cast

import pytest
from pydantic import (
    BaseModel,
    ConfigDict,
    PrivateAttr,
    field_validator,
    model_serializer,
    model_validator,
)

from agents.exceptions import (
    BaseExceptionGroup,
    _mark_error_data_redacted,
    _raise_data_redacted_error,
)
from agents.extensions.sandbox.blaxel.mounts import BlaxelCloudBucketMountStrategy
from agents.extensions.sandbox.daytona.mounts import DaytonaCloudBucketMountStrategy
from agents.extensions.sandbox.e2b.mounts import E2BCloudBucketMountStrategy
from agents.extensions.sandbox.modal.mounts import ModalCloudBucketMountStrategy
from agents.extensions.sandbox.runloop.mounts import RunloopCloudBucketMountStrategy
from agents.sandbox import Manifest
from agents.sandbox._mount_security import (
    CREDENTIALLESS_MOUNT_AUTHORITY_KEY,
    REDACTED_MOUNT_AUTHORITY_KEY,
    redact_mount_error_data,
    redact_mount_error_data_sync,
    sanitize_manifest_mount_authority,
    sanitize_raw_session_state_mount_authority,
    validate_manifest_mount_credential_boundaries,
    validate_mount_activation_credential_boundary,
)
from agents.sandbox.entries import (
    AzureBlobMount,
    BaseEntry,
    BoxMount,
    Dir,
    DockerVolumeMountStrategy,
    File,
    FuseMountPattern,
    GCSMount,
    GitRepo,
    InContainerMountStrategy,
    LocalDir,
    LocalFile,
    Mount,
    MountpointMountPattern,
    MountStrategyBase,
    R2Mount,
    RcloneMountPattern,
    S3FilesMount,
    S3FilesMountPattern,
    S3Mount,
)
from agents.sandbox.entries.mounts.base import InContainerMountAdapter
from agents.sandbox.entries.mounts.patterns import (
    MountPattern,
    MountPatternConfig,
    RcloneMountConfig,
)
from agents.sandbox.errors import (
    ErrorCode,
    MountCommandError,
    MountConfigError,
    MountToolMissingError,
    SandboxError,
)
from agents.sandbox.manifest import Environment
from agents.sandbox.session.sandbox_client import BaseSandboxClient
from agents.sandbox.session.sandbox_session import SandboxSession
from agents.sandbox.session.sandbox_session_state import SandboxSessionState
from agents.sandbox.snapshot import NoopSnapshot, SnapshotBase, SnapshotSpec
from tests.utils.factories import TestSessionState


class _SecurityTestClient(BaseSandboxClient[None]):
    backend_id = "test"

    async def create(
        self,
        *,
        snapshot: SnapshotSpec | SnapshotBase | None = None,
        manifest: Manifest | None = None,
        options: None,
    ) -> SandboxSession:
        _ = (snapshot, manifest, options)
        raise AssertionError("create() is not used in these tests")

    async def delete(self, session: SandboxSession) -> SandboxSession:
        raise AssertionError(f"delete() is not used in these tests: {session!r}")

    async def resume(self, state: SandboxSessionState) -> SandboxSession:
        raise AssertionError(f"resume() is not used in these tests: {state!r}")

    def deserialize_session_state(self, payload: dict[str, object]) -> SandboxSessionState:
        return self._deserialize_session_state_payload(payload, TestSessionState)


class _CustomTokenEntry(BaseEntry):
    type: Literal["custom_token_entry"] = "custom_token_entry"
    token: str

    async def apply(self, session: Any, dest: Path, base_dir: Path) -> list[Any]:
        _ = (session, dest, base_dir)
        return []


def _install_hostile_exception_descriptors(error_type: type[BaseException]) -> None:
    def get_base_args(error: BaseException) -> tuple[object, ...]:
        return cast(
            tuple[object, ...],
            cast(Any, BaseException.args).__get__(error, type(error)),
        )

    def reject_traceback_access(error: BaseException) -> object:
        _ = error
        raise AssertionError("provider-defined traceback descriptor was accessed")

    type.__setattr__(error_type, "args", property(get_base_args))
    type.__setattr__(error_type, "__traceback__", property(reject_traceback_access))


def _assert_sanitized_mount_command_error(error: BaseException) -> None:
    assert type(error) is MountCommandError
    assert error.error_code is ErrorCode.MOUNT_FAILED
    assert error.op == "materialize"
    assert error.retryable is False
    assert error.context == {"command": "<redacted>", "stderr": None}
    assert str(error) == "mount command failed"
    assert error.__cause__ is None
    assert error.__context__ is None


def _assert_sanitized_mount_tool_missing_error(error: BaseException) -> None:
    assert type(error) is MountToolMissingError
    assert error.error_code is ErrorCode.MOUNT_MISSING_TOOL
    assert error.op == "materialize"
    assert error.retryable is False
    assert error.context == {"tool": "<redacted>"}
    assert str(error) == "required mount tool missing: <redacted>"
    assert error.__cause__ is None
    assert error.__context__ is None


class _CustomChildrenEntry(BaseEntry):
    type: Literal["custom_children_entry"] = "custom_children_entry"
    children: Any

    async def apply(self, session: Any, dest: Path, base_dir: Path) -> list[Any]:
        _ = (session, dest, base_dir)
        return []


class _CustomCredentialSourceEntry(BaseEntry):
    type: Literal["custom_credential_source_entry"] = "custom_credential_source_entry"
    content: str
    source_token: str

    async def apply(self, session: Any, dest: Path, base_dir: Path) -> list[Any]:
        _ = (session, dest, base_dir)
        return []


class _DirectCustomMount(Mount):
    type: Literal["direct_custom_mount"] = "direct_custom_mount"
    bucket: str
    api_token: str

    def in_container_adapter(self) -> InContainerMountAdapter:
        return InContainerMountAdapter(self)

    def supported_in_container_patterns(
        self,
    ) -> tuple[builtins.type[RcloneMountPattern], ...]:
        return (RcloneMountPattern,)

    def supported_docker_volume_drivers(self) -> frozenset[str]:
        return frozenset({"rclone"})

    async def build_in_container_mount_config(
        self,
        session: Any,
        pattern: MountPattern,
        *,
        include_config_text: bool,
    ) -> MountPatternConfig:
        _ = (session, pattern, include_config_text)
        return RcloneMountConfig(
            remote_name="direct-custom",
            remote_path=self.bucket,
            remote_kind="s3",
            mount_type=self.type,
            config_text=f"api_token = {self.api_token}\n",
        )


class _CustomPatternStrategy(MountStrategyBase):
    type: Literal["custom_pattern_strategy"] = "custom_pattern_strategy"
    pattern: dict[str, Any]
    api_token: str | None = None

    def validate_mount(self, mount: Any) -> None:
        _ = mount

    async def activate(self, mount: Any, session: Any, dest: Path, base_dir: Path) -> list[Any]:
        _ = (mount, session, dest, base_dir)
        return []

    async def deactivate(self, mount: Any, session: Any, dest: Path, base_dir: Path) -> None:
        _ = (mount, session, dest, base_dir)

    async def teardown_for_snapshot(self, mount: Any, session: Any, path: Path) -> None:
        _ = (mount, session, path)

    async def restore_after_snapshot(self, mount: Any, session: Any, path: Path) -> None:
        _ = (mount, session, path)

    def build_docker_volume_driver_config(
        self, mount: Any
    ) -> tuple[str, dict[str, str], bool] | None:
        _ = mount
        return None


class _CustomInContainerStrategy(InContainerMountStrategy):
    type: Literal["custom_in_container_strategy"] = "custom_in_container_strategy"  # type: ignore[assignment]


class _CustomDockerVolumeStrategy(DockerVolumeMountStrategy):
    type: Literal["custom_docker_volume_strategy"] = "custom_docker_volume_strategy"  # type: ignore[assignment]


class _CustomModalCloudBucketStrategy(ModalCloudBucketMountStrategy):
    type: Literal["custom_modal_cloud_bucket_strategy"] = "custom_modal_cloud_bucket_strategy"  # type: ignore[assignment]


def _s3_mount(
    *,
    strategy: InContainerMountStrategy | DockerVolumeMountStrategy,
    credentialed: bool = False,
) -> S3Mount:
    return S3Mount(
        bucket="example-bucket",
        access_key_id="example-access-key" if credentialed else None,
        secret_access_key="example-secret-key" if credentialed else None,
        mount_strategy=strategy,
    )


def test_rejects_explicit_credentials_for_in_container_mounts() -> None:
    manifest = Manifest(
        entries={
            "data": _s3_mount(
                strategy=InContainerMountStrategy(pattern=RcloneMountPattern()),
                credentialed=True,
            )
        }
    )

    with pytest.raises(MountConfigError, match="mount-scoped credentials") as exc:
        validate_manifest_mount_credential_boundaries(manifest)

    assert exc.value.context["credential_fields"] == (
        "access_key_id",
        "secret_access_key",
    )
    assert "example-secret-key" not in str(exc.value)
    assert "example-secret-key" not in repr(exc.value.context)


def test_exact_path_acknowledgement_allows_supported_mount_scoped_credentials() -> None:
    manifest = Manifest(
        entries={
            "data": _s3_mount(
                strategy=InContainerMountStrategy(pattern=RcloneMountPattern()),
                credentialed=True,
            )
        }
    ).with_in_container_mount_credential_exposure_acknowledged("data")

    validate_manifest_mount_credential_boundaries(manifest)

    sibling = manifest.model_copy(deep=True)
    sibling.entries["other"] = sibling.entries.pop("data")
    with pytest.raises(MountConfigError, match="mount-scoped credentials"):
        validate_manifest_mount_credential_boundaries(sibling)

    mount = manifest.entries["data"]
    assert isinstance(mount, S3Mount)
    validate_mount_activation_credential_boundary(
        mount,
        mount.mount_strategy,
        manifest=manifest,
        mount_path="/workspace/data",
        provider_backend_id="docker",
    )
    with pytest.raises(MountConfigError, match="mount-scoped credentials"):
        validate_mount_activation_credential_boundary(
            mount,
            mount.mount_strategy,
            manifest=manifest,
            mount_path="/workspace/other",
            provider_backend_id="docker",
        )


@pytest.mark.parametrize(
    ("credentials", "invalid_fields"),
    [
        ({"access_key_id": "access-key"}, ("secret_access_key",)),
        ({"secret_access_key": "secret-key"}, ("access_key_id",)),
        (
            {"session_token": "session-token"},
            ("access_key_id", "secret_access_key"),
        ),
        (
            {"access_key_id": "access-key", "secret_access_key": ""},
            ("secret_access_key",),
        ),
        (
            {"access_key_id": " ", "secret_access_key": "secret-key"},
            ("access_key_id",),
        ),
        (
            {
                "access_key_id": "access-key",
                "secret_access_key": "secret-key",
                "session_token": " ",
            },
            ("session_token",),
        ),
    ],
)
def test_acknowledgement_rejects_incomplete_in_container_s3_credentials(
    credentials: dict[str, str],
    invalid_fields: tuple[str, ...],
) -> None:
    manifest = Manifest(
        entries={
            "data": S3Mount(
                bucket="example-bucket",
                mount_strategy=InContainerMountStrategy(pattern=RcloneMountPattern()),
                **cast(Any, credentials),
            )
        }
    ).with_in_container_mount_credential_exposure_acknowledged("data")

    with pytest.raises(MountConfigError, match="complete non-empty credential set") as exc_info:
        validate_manifest_mount_credential_boundaries(manifest)

    assert exc_info.value.context["credential_fields"] == invalid_fields


@pytest.mark.parametrize(
    ("credentials", "invalid_fields"),
    [
        ({"access_id": "access-id"}, ("secret_access_key",)),
        ({"secret_access_key": "secret-key"}, ("access_id",)),
        (
            {"access_id": "access-id", "secret_access_key": ""},
            ("secret_access_key",),
        ),
        (
            {"access_id": " ", "secret_access_key": "secret-key"},
            ("access_id",),
        ),
        (
            {
                "access_id": "access-id",
                "service_account_credentials": '{"type":"service_account"}',
            },
            ("secret_access_key",),
        ),
    ],
)
def test_acknowledgement_rejects_incomplete_in_container_gcs_hmac_credentials(
    credentials: dict[str, str],
    invalid_fields: tuple[str, ...],
) -> None:
    manifest = Manifest(
        entries={
            "data": GCSMount(
                bucket="example-bucket",
                mount_strategy=InContainerMountStrategy(pattern=RcloneMountPattern()),
                **cast(Any, credentials),
            )
        }
    ).with_in_container_mount_credential_exposure_acknowledged("data")

    with pytest.raises(MountConfigError, match="complete non-empty credential set") as exc_info:
        validate_manifest_mount_credential_boundaries(manifest)

    assert exc_info.value.context["credential_fields"] == invalid_fields


def test_acknowledgement_accepts_complete_in_container_gcs_hmac_credentials() -> None:
    manifest = Manifest(
        entries={
            "data": GCSMount(
                bucket="example-bucket",
                access_id="access-id",
                secret_access_key="secret-key",
                mount_strategy=InContainerMountStrategy(pattern=RcloneMountPattern()),
            )
        }
    ).with_in_container_mount_credential_exposure_acknowledged("data")

    validate_manifest_mount_credential_boundaries(manifest)


@pytest.mark.parametrize("blank_value", ["", "   "])
@pytest.mark.parametrize(
    ("mount_factory", "broad", "invalid_field"),
    [
        (
            lambda value: GCSMount(
                bucket="example-bucket",
                access_token=value,
                mount_strategy=InContainerMountStrategy(pattern=RcloneMountPattern()),
            ),
            False,
            "access_token",
        ),
        (
            lambda value: GCSMount(
                bucket="example-bucket",
                service_account_credentials=value,
                mount_strategy=InContainerMountStrategy(pattern=RcloneMountPattern()),
            ),
            False,
            "service_account_credentials",
        ),
        (
            lambda value: GCSMount(
                bucket="example-bucket",
                service_account_file=value,
                mount_strategy=InContainerMountStrategy(pattern=RcloneMountPattern()),
            ),
            True,
            "service_account_file",
        ),
        (
            lambda value: AzureBlobMount(
                account="example-account",
                container="example-container",
                account_key=value,
                mount_strategy=InContainerMountStrategy(pattern=RcloneMountPattern()),
            ),
            False,
            "account_key",
        ),
        (
            lambda value: AzureBlobMount(
                account="example-account",
                container="example-container",
                identity_client_id=value,
                mount_strategy=InContainerMountStrategy(pattern=RcloneMountPattern()),
            ),
            True,
            "identity_client_id",
        ),
    ],
)
def test_acknowledgement_rejects_empty_in_container_scalar_authority(
    mount_factory: Any,
    broad: bool,
    invalid_field: str,
    blank_value: str,
) -> None:
    manifest = Manifest(entries={"data": mount_factory(blank_value)})
    acknowledged = (
        manifest.with_in_container_mount_broad_credential_exposure_acknowledged("data")
        if broad
        else manifest.with_in_container_mount_credential_exposure_acknowledged("data")
    )

    with pytest.raises(MountConfigError, match="must not be empty or whitespace-only") as exc_info:
        validate_manifest_mount_credential_boundaries(acknowledged)

    assert exc_info.value.context["credential_fields"] == (invalid_field,)


@pytest.mark.parametrize(
    ("credentials", "invalid_fields"),
    [
        ({"access_key_id": "access-key"}, ("secret_access_key",)),
        ({"secret_access_key": "secret-key"}, ("access_key_id",)),
        (
            {"access_key_id": "access-key", "secret_access_key": ""},
            ("secret_access_key",),
        ),
    ],
)
def test_acknowledgement_rejects_incomplete_in_container_r2_credentials(
    credentials: dict[str, str],
    invalid_fields: tuple[str, ...],
) -> None:
    manifest = Manifest(
        entries={
            "data": R2Mount(
                bucket="example-bucket",
                account_id="example-account",
                mount_strategy=InContainerMountStrategy(pattern=RcloneMountPattern()),
                **cast(Any, credentials),
            )
        }
    ).with_in_container_mount_credential_exposure_acknowledged("data")

    with pytest.raises(MountConfigError, match="complete non-empty credential set") as exc_info:
        validate_manifest_mount_credential_boundaries(manifest)

    assert exc_info.value.context["credential_fields"] == invalid_fields


@pytest.mark.parametrize(
    "mount",
    [
        S3Mount(
            bucket="example-bucket",
            access_key_id="access-key",
            mount_strategy=DockerVolumeMountStrategy(driver="rclone"),
        ),
        GCSMount(
            bucket="example-bucket",
            access_id="access-id",
            mount_strategy=DockerVolumeMountStrategy(driver="rclone"),
        ),
        R2Mount(
            bucket="example-bucket",
            account_id="example-account",
            access_key_id="access-key",
            mount_strategy=DockerVolumeMountStrategy(driver="rclone"),
        ),
        GCSMount(
            bucket="example-bucket",
            access_token="",
            mount_strategy=DockerVolumeMountStrategy(driver="rclone"),
        ),
        AzureBlobMount(
            account="example-account",
            container="example-container",
            identity_client_id=" ",
            mount_strategy=DockerVolumeMountStrategy(driver="rclone"),
        ),
    ],
)
def test_incomplete_credentials_remain_external_provider_configuration(mount: Mount) -> None:
    manifest = Manifest(entries={"data": mount})

    validate_manifest_mount_credential_boundaries(manifest, provider_backend_id="docker")


def test_mount_credential_acknowledgement_is_not_a_path_prefix() -> None:
    mount = _s3_mount(
        strategy=InContainerMountStrategy(pattern=RcloneMountPattern()),
        credentialed=True,
    )
    manifest = Manifest(
        entries={"parent": Dir(children={"data": mount})}
    ).with_in_container_mount_credential_exposure_acknowledged("parent")

    with pytest.raises(MountConfigError, match="mount-scoped credentials"):
        validate_manifest_mount_credential_boundaries(manifest)

    validate_manifest_mount_credential_boundaries(
        manifest.with_in_container_mount_credential_exposure_acknowledged("parent/data")
    )


def test_mount_credential_acknowledgement_preserves_path_whitespace() -> None:
    manifest = Manifest(
        entries={
            "data ": _s3_mount(
                strategy=InContainerMountStrategy(pattern=RcloneMountPattern()),
                credentialed=True,
            )
        }
    ).with_in_container_mount_credential_exposure_acknowledged("data")

    with pytest.raises(MountConfigError, match="mount-scoped credentials"):
        validate_manifest_mount_credential_boundaries(manifest)

    validate_manifest_mount_credential_boundaries(
        manifest.with_in_container_mount_credential_exposure_acknowledged("data ")
    )


def test_mount_credential_acknowledgement_accepts_platform_path_objects() -> None:
    manifest = Manifest(
        entries={
            "data": _s3_mount(
                strategy=InContainerMountStrategy(pattern=RcloneMountPattern()),
                credentialed=True,
            )
        }
    ).with_in_container_mount_credential_exposure_acknowledged(PureWindowsPath("data"))

    mount = manifest.entries["data"]
    assert isinstance(mount, S3Mount)
    validate_mount_activation_credential_boundary(
        mount,
        mount.mount_strategy,
        manifest=manifest,
        mount_path=PureWindowsPath("/workspace/data"),
        provider_backend_id="docker",
    )
    assert not Manifest()._acknowledges_in_container_mount_credential_exposure(
        PureWindowsPath("/workspace/data"),
        "mount_scoped",
    )

    with pytest.raises(ValueError, match="use '/' separators"):
        Manifest().with_in_container_mount_credential_exposure_acknowledged("data\\child")


@pytest.mark.parametrize(
    "policy_key",
    [
        "in_container_mount_credential_exposure_acknowledged_paths",
        "_in_container_mount_credential_exposure_acknowledged_paths",
        "inContainerMountCredentialExposureAcknowledgedPaths",
        "in_container_mount_broad_credential_exposure_acknowledged_paths",
        "_mount_credential_exposure_policy",
    ],
)
def test_manifest_input_cannot_inject_mount_credential_acknowledgement(policy_key: str) -> None:
    with pytest.raises(TypeError, match="trusted Manifest instance"):
        Manifest.model_validate({policy_key: ["data"]})


def test_manifest_acknowledgement_is_runtime_only_and_rejects_root() -> None:
    manifest = Manifest().with_in_container_mount_credential_exposure_acknowledged("data")
    payload = manifest.model_dump(mode="json")

    assert all("credential_exposure" not in key for key in payload)
    restored = Manifest.model_validate(payload)
    assert not restored._acknowledges_in_container_mount_credential_exposure(
        "/workspace/data", "mount_scoped"
    )
    with pytest.raises(ValueError, match="non-root path"):
        Manifest().with_in_container_mount_credential_exposure_acknowledged("/workspace")
    with pytest.raises(TypeError, match="At least one"):
        Manifest().with_in_container_mount_credential_exposure_acknowledged()


@pytest.mark.parametrize(
    "method_name",
    [
        "with_in_container_mount_credential_exposure_acknowledged",
        "with_in_container_mount_broad_credential_exposure_acknowledged",
    ],
)
@pytest.mark.parametrize(
    "path",
    [
        "data/*",
        "data?",
        "data[0]",
        "data/../other",
        "/workspace/../outside",
    ],
)
def test_manifest_acknowledgement_rejects_wildcard_and_parent_paths(
    method_name: str,
    path: str,
) -> None:
    method = getattr(Manifest(), method_name)

    with pytest.raises(ValueError, match="wildcard syntax|parent segments"):
        method(path)


def test_manifest_acknowledgement_rejects_custom_mount_before_deepcopy() -> None:
    sentinel = "custom-mount-deepcopy-secret"

    class CustomS3Mount(S3Mount):
        type: Literal["custom_deepcopy_s3_mount"] = "custom_deepcopy_s3_mount"  # type: ignore[assignment]
        deepcopy_called: ClassVar[bool] = False

        def __deepcopy__(self, memo: dict[int, Any] | None = None) -> CustomS3Mount:
            _ = memo
            type(self).deepcopy_called = True
            raise RuntimeError(sentinel)

    manifest = Manifest(
        entries={
            "data": CustomS3Mount(
                bucket="bucket",
                mount_strategy=InContainerMountStrategy(pattern=RcloneMountPattern()),
            )
        }
    )

    with pytest.raises(
        MountConfigError, match="sandbox mount configuration is invalid"
    ) as exc_info:
        manifest.with_in_container_mount_credential_exposure_acknowledged("data")

    assert CustomS3Mount.deepcopy_called is False
    assert sentinel not in repr(exc_info.value)


def test_manifest_acknowledgement_redacts_custom_provenance_traceback_locals() -> None:
    sentinel = "custom-provenance-traceback-secret"

    class CustomS3Mount(S3Mount):
        type: Literal["custom_traceback_s3_mount"] = "custom_traceback_s3_mount"  # type: ignore[assignment]
        api_token: str | None = None

    class CustomInContainerStrategy(InContainerMountStrategy):
        type: Literal["custom_traceback_strategy"] = "custom_traceback_strategy"  # type: ignore[assignment]
        api_token: str | None = None

    class CustomRclonePattern(RcloneMountPattern):
        api_token: str | None = None

    cases = [
        (
            Manifest(
                entries={
                    "data": CustomS3Mount(
                        bucket="bucket",
                        api_token=sentinel,
                        mount_strategy=InContainerMountStrategy(pattern=RcloneMountPattern()),
                    )
                }
            ),
            "custom mount implementations",
        ),
        (
            Manifest(
                entries={
                    "data": S3Mount(
                        bucket="bucket",
                        mount_strategy=CustomInContainerStrategy(
                            api_token=sentinel,
                            pattern=RcloneMountPattern(),
                        ),
                    )
                }
            ),
            "custom mount strategies",
        ),
        (
            Manifest(
                entries={
                    "data": S3Mount(
                        bucket="bucket",
                        mount_strategy=InContainerMountStrategy(
                            pattern=CustomRclonePattern(api_token=sentinel)
                        ),
                    )
                }
            ),
            "custom mount patterns",
        ),
    ]

    for method_name in (
        "with_in_container_mount_credential_exposure_acknowledged",
        "with_in_container_mount_broad_credential_exposure_acknowledged",
    ):
        for manifest, _message in cases:
            method = getattr(manifest, method_name)
            with pytest.raises(
                MountConfigError,
                match="sandbox mount configuration is invalid",
            ) as exc:
                method("data")

            traceback_cursor = exc.value.__traceback__
            while traceback_cursor is not None:
                module_name = str(traceback_cursor.tb_frame.f_globals.get("__name__", ""))
                if module_name.startswith("agents."):
                    assert sentinel not in repr(traceback_cursor.tb_frame.f_locals)
                traceback_cursor = traceback_cursor.tb_next


def test_builtin_mount_subclass_is_rejected_by_execution_provenance() -> None:
    class CustomS3Mount(S3Mount):
        type: Literal["custom_s3_mount"] = "custom_s3_mount"  # type: ignore[assignment]
        api_token: str | None = None

        def _rclone_required_lines(self, remote_name: str) -> list[str]:
            lines = super()._rclone_required_lines(remote_name)
            if self.api_token is not None:
                lines.append(f"api_token = {self.api_token}")
            return lines

    in_container = Manifest(
        entries={
            "data": CustomS3Mount(
                bucket="example-bucket",
                api_token="custom-mount-secret",
                mount_strategy=InContainerMountStrategy(pattern=RcloneMountPattern()),
            )
        }
    )

    with pytest.raises(MountConfigError, match="custom mount implementations"):
        validate_manifest_mount_credential_boundaries(in_container)

    external = Manifest(
        entries={
            "data": CustomS3Mount(
                bucket="example-bucket",
                access_key_id="example-access-key",
                secret_access_key="example-secret-key",
                api_token="custom-mount-secret",
                mount_strategy=DockerVolumeMountStrategy(driver="rclone"),
            )
        }
    )
    with pytest.raises(MountConfigError, match="custom mount implementations") as exc_info:
        sanitize_manifest_mount_authority(external)

    assert "example-secret-key" not in repr(exc_info.value)
    assert "custom-mount-secret" not in repr(exc_info.value)


def test_builtin_mount_subclass_rejects_pydantic_extra_configuration() -> None:
    class ExtraS3Mount(S3Mount):
        type: Literal["extra_s3_mount"] = "extra_s3_mount"  # type: ignore[assignment]
        model_config = ConfigDict(extra="allow")

        def _rclone_required_lines(self, remote_name: str) -> list[str]:
            lines = super()._rclone_required_lines(remote_name)
            lines.append(f"api_token = {cast(Any, self).api_token}")
            return lines

    mount = ExtraS3Mount.model_validate(
        {
            "bucket": "example-bucket",
            "api_token": "custom-mount-extra-secret",
            "mount_strategy": InContainerMountStrategy(pattern=RcloneMountPattern()),
        }
    )

    with pytest.raises(MountConfigError, match="custom mount implementations") as exc:
        validate_manifest_mount_credential_boundaries(Manifest(entries={"data": mount}))

    assert "custom-mount-extra-secret" not in str(exc.value)


def test_direct_custom_mount_configuration_is_opaque_authority() -> None:
    sentinel = "direct-custom-mount-secret"
    mount = _DirectCustomMount(
        bucket="bucket",
        api_token=sentinel,
        mount_strategy=InContainerMountStrategy(pattern=RcloneMountPattern()),
    )

    with pytest.raises(MountConfigError, match="custom mount implementations") as exc:
        validate_manifest_mount_credential_boundaries(Manifest(entries={"data": mount}))

    assert sentinel not in str(exc.value)


def test_behavior_only_mount_subclass_is_rejected_before_config_generation() -> None:
    class BehaviorOnlyS3Mount(S3Mount):
        type: Literal["behavior_only_s3_mount"] = "behavior_only_s3_mount"  # type: ignore[assignment]
        config_called: ClassVar[bool] = False

        def _rclone_required_lines(self, remote_name: str) -> list[str]:
            type(self).config_called = True
            return [f"[{remote_name}]", "type = s3", "env_auth = true"]

    mount = BehaviorOnlyS3Mount(
        bucket="bucket",
        mount_strategy=InContainerMountStrategy(pattern=RcloneMountPattern()),
    )

    with pytest.raises(MountConfigError, match="custom mount implementations"):
        validate_manifest_mount_credential_boundaries(Manifest(entries={"data": mount}))

    assert BehaviorOnlyS3Mount.config_called is False


def test_custom_mount_is_rejected_before_mount_path_resolution() -> None:
    sentinel = "custom-mount-path-secret"

    class CustomPathS3Mount(S3Mount):
        type: Literal["custom_path_s3_mount"] = "custom_path_s3_mount"  # type: ignore[assignment]
        _private_authority: str = PrivateAttr(default=sentinel)
        resolver_called: ClassVar[bool] = False

        def _resolve_mount_path_for_root(self, root: Path, dest: Path) -> Path:
            _ = (root, dest)
            type(self).resolver_called = True
            raise RuntimeError(self._private_authority)

    mount = CustomPathS3Mount(
        bucket="bucket",
        mount_strategy=InContainerMountStrategy(pattern=RcloneMountPattern()),
    )

    with pytest.raises(MountConfigError, match="custom mount implementations") as exc_info:
        validate_manifest_mount_credential_boundaries(Manifest(entries={"data": mount}))

    assert CustomPathS3Mount.resolver_called is False
    assert sentinel not in repr(exc_info.value)


def test_custom_mount_cannot_self_declare_a_trusted_credential_boundary() -> None:
    sentinel = "private-custom-secret"

    class SelfDeclaredTrustedS3Mount(S3Mount):
        type: Literal["self_declared_trusted_s3_mount"] = "self_declared_trusted_s3_mount"  # type: ignore[assignment]
        _trusted_application_credential_boundary: ClassVar[bool] = True
        _private_credential: str = PrivateAttr(default=sentinel)
        config_called: ClassVar[bool] = False

        def _rclone_required_lines(self, remote_name: str) -> list[str]:
            type(self).config_called = True
            return [
                f"[{remote_name}]",
                "type = s3",
                f"secret_access_key = {self._private_credential}",
            ]

    mount = SelfDeclaredTrustedS3Mount(
        bucket="public-bucket",
        mount_strategy=InContainerMountStrategy(pattern=RcloneMountPattern()),
    )

    with pytest.raises(MountConfigError, match="custom mount implementations") as exc_info:
        validate_manifest_mount_credential_boundaries(Manifest(entries={"data": mount}))

    assert sentinel not in str(exc_info.value)
    assert SelfDeclaredTrustedS3Mount.config_called is False


def test_direct_custom_mount_configuration_cannot_enter_durable_state() -> None:
    sentinel = "direct-custom-durable-secret"
    state = TestSessionState(
        manifest=Manifest(
            entries={
                "data": _DirectCustomMount(
                    bucket="bucket",
                    api_token=sentinel,
                    mount_strategy=DockerVolumeMountStrategy(driver="rclone"),
                )
            }
        ),
        snapshot=NoopSnapshot(id="snapshot"),
    )

    with pytest.raises(MountConfigError) as exc:
        _SecurityTestClient().serialize_session_state(state)

    assert sentinel not in str(exc.value)
    assert sentinel not in repr(exc.value)
    traceback = exc.value.__traceback__
    while traceback is not None:
        frame_path = Path(traceback.tb_frame.f_code.co_filename).as_posix()
        if "/src/agents/" in frame_path:
            assert sentinel not in repr(traceback.tb_frame.f_locals)
        traceback = traceback.tb_next


def test_custom_mount_is_rejected_before_durable_serializer_runs() -> None:
    sentinel = "custom-mount-serializer-secret"

    class CustomSerializedS3Mount(S3Mount):
        type: Literal["custom_serialized_s3_mount"] = "custom_serialized_s3_mount"  # type: ignore[assignment]
        _private_authority: str = PrivateAttr(default=sentinel)
        serializer_called: ClassVar[bool] = False

        @model_serializer(mode="wrap")
        def _serialize(self, handler: Any) -> Any:
            _ = handler
            type(self).serializer_called = True
            raise RuntimeError(self._private_authority)

    state = TestSessionState(
        manifest=Manifest(
            entries={
                "data": CustomSerializedS3Mount(
                    bucket="bucket",
                    mount_strategy=DockerVolumeMountStrategy(driver="rclone"),
                )
            }
        ),
        snapshot=NoopSnapshot(id="snapshot"),
    )

    with pytest.raises(MountConfigError) as exc_info:
        _SecurityTestClient().serialize_session_state(state)

    assert CustomSerializedS3Mount.serializer_called is False
    assert sentinel not in repr(exc_info.value)


def test_public_mount_error_redactor_discards_untrusted_mount_discriminator() -> None:
    sentinel = "custom-mount-type-secret"

    class CustomS3Mount(S3Mount):
        type: Literal["custom-mount-type-secret"] = sentinel  # type: ignore[assignment]
        api_token: str | None = None

    manifest = Manifest(
        entries={
            "data": CustomS3Mount(
                bucket="example-bucket",
                api_token="configured",
                mount_strategy=InContainerMountStrategy(pattern=RcloneMountPattern()),
            )
        }
    )

    @redact_mount_error_data_sync
    def validate(*, manifest: Manifest) -> None:
        validate_manifest_mount_credential_boundaries(manifest)

    with pytest.raises(MountConfigError) as exc:
        validate(manifest=manifest)

    assert sentinel not in repr(exc.value)


def test_public_mount_error_redactor_discards_untrusted_field_names() -> None:
    sentinel = "custom-mount-field-secret"

    class ExtraS3Mount(S3Mount):
        type: Literal["custom_extra_s3_mount"] = "custom_extra_s3_mount"  # type: ignore[assignment]
        model_config = ConfigDict(extra="allow")

    mount = ExtraS3Mount.model_validate(
        {
            "bucket": "example-bucket",
            sentinel: "configured",
            "mount_strategy": InContainerMountStrategy(pattern=RcloneMountPattern()),
        }
    )
    manifest = Manifest(entries={"data": mount})

    @redact_mount_error_data_sync
    def validate(*, manifest: Manifest) -> None:
        validate_manifest_mount_credential_boundaries(manifest)

    with pytest.raises(MountConfigError) as exc:
        validate(manifest=manifest)

    assert sentinel not in repr(exc.value)


def test_public_mount_error_redactor_rejects_before_custom_attribute_access() -> None:
    class CustomS3Mount(S3Mount):
        type: Literal["custom_attribute_s3_mount"] = "custom_attribute_s3_mount"  # type: ignore[assignment]
        authority_accessed: ClassVar[bool] = False

        def __getattribute__(self, name: str) -> Any:
            if name == "access_key_id":
                type(self).authority_accessed = True
            return super().__getattribute__(name)

    mount = CustomS3Mount(
        bucket="example-bucket",
        access_key_id="access-key",
        secret_access_key="custom-attribute-secret",
        mount_strategy=InContainerMountStrategy(pattern=RcloneMountPattern()),
    )
    CustomS3Mount.authority_accessed = False
    manifest = Manifest(entries={"data": mount})

    @redact_mount_error_data_sync
    def validate(*, manifest: Manifest) -> None:
        validate_manifest_mount_credential_boundaries(manifest)

    with pytest.raises(MountConfigError):
        validate(manifest=manifest)

    assert CustomS3Mount.authority_accessed is False


def test_rejection_redacts_sdk_traceback_frames_without_mutating_trusted_manifest() -> None:
    sentinel = "traceback-secret"
    manifest = Manifest(
        entries={
            "data": S3Mount(
                bucket="example-bucket",
                access_key_id="access-key",
                secret_access_key=sentinel,
                mount_strategy=InContainerMountStrategy(pattern=RcloneMountPattern()),
            )
        }
    )

    with pytest.raises(MountConfigError) as exc:
        validate_manifest_mount_credential_boundaries(manifest)

    mount = manifest.entries["data"]
    assert isinstance(mount, S3Mount)
    assert mount.access_key_id == "access-key"
    assert mount.secret_access_key == sentinel
    assert exc.value.__cause__ is None
    assert exc.value.__context__ is None
    traceback = exc.value.__traceback__
    while traceback is not None:
        module_name = traceback.tb_frame.f_globals.get("__name__", "")
        if isinstance(module_name, str) and module_name.startswith("agents."):
            assert sentinel not in repr(traceback.tb_frame.f_locals)
        traceback = traceback.tb_next


@pytest.mark.asyncio
async def test_authority_detection_keeps_invalid_manifest_paths_inside_redaction_boundary() -> None:
    sentinel = "invalid-path-secret"
    manifest = Manifest(
        entries={
            "../data": S3Mount(
                bucket="bucket",
                access_key_id="access-key",
                secret_access_key=sentinel,
                mount_strategy=DockerVolumeMountStrategy(driver="rclone"),
            )
        }
    )

    @redact_mount_error_data
    async def validate(*, manifest: Manifest) -> None:
        validate_manifest_mount_credential_boundaries(manifest)

    with pytest.raises(SandboxError, match="protected mount configuration") as exc:
        await validate(manifest=manifest)

    assert sentinel not in str(exc.value)
    traceback_cursor = exc.value.__traceback__
    while traceback_cursor is not None:
        module_name = traceback_cursor.tb_frame.f_globals.get("__name__", "")
        if isinstance(module_name, str) and module_name.startswith("agents."):
            assert sentinel not in repr(traceback_cursor.tb_frame.f_locals)
        traceback_cursor = traceback_cursor.tb_next


def test_preserves_credentialless_in_container_and_credentialed_docker_mounts() -> None:
    anonymous = Manifest(
        entries={"data": _s3_mount(strategy=InContainerMountStrategy(pattern=RcloneMountPattern()))}
    )
    docker = Manifest(
        entries={
            "data": _s3_mount(
                strategy=DockerVolumeMountStrategy(
                    driver="rclone",
                    driver_options={"vfs-cache-mode": "off"},
                ),
                credentialed=True,
            )
        }
    )

    validate_manifest_mount_credential_boundaries(anonymous)
    validate_manifest_mount_credential_boundaries(docker, provider_backend_id="docker")


@pytest.mark.parametrize(
    ("backend_id", "strategy"),
    [
        ("blaxel", BlaxelCloudBucketMountStrategy()),
        ("daytona", DaytonaCloudBucketMountStrategy()),
        ("e2b", E2BCloudBucketMountStrategy()),
        ("runloop", RunloopCloudBucketMountStrategy()),
    ],
)
def test_preserves_credentialless_hosted_mount_strategies(
    backend_id: str,
    strategy: MountStrategyBase,
) -> None:
    manifest = Manifest(entries={"data": S3Mount(bucket="example-bucket", mount_strategy=strategy)})

    validate_manifest_mount_credential_boundaries(
        manifest,
        provider_backend_id=backend_id,
    )

    credentialed = manifest.model_copy(deep=True)
    mount = credentialed.entries["data"]
    assert isinstance(mount, S3Mount)
    mount.access_key_id = "example-access-key"
    mount.secret_access_key = "example-secret-key"
    with pytest.raises(MountConfigError, match="mount-scoped credentials"):
        validate_manifest_mount_credential_boundaries(
            credentialed,
            provider_backend_id=backend_id,
        )


def test_custom_strategy_cannot_declare_itself_external() -> None:
    class ForgedExternalStrategy(InContainerMountStrategy):
        type: Literal["forged_external"] = "forged_external"  # type: ignore[assignment]
        _credential_boundary: ClassVar[str] = "external"

    manifest = Manifest(
        entries={
            "data": _s3_mount(
                strategy=ForgedExternalStrategy(pattern=RcloneMountPattern()),
                credentialed=True,
            )
        }
    )

    with pytest.raises(MountConfigError, match="custom mount strategies"):
        validate_manifest_mount_credential_boundaries(manifest, provider_backend_id="unix_local")


@pytest.mark.parametrize(
    "strategy",
    [
        _CustomDockerVolumeStrategy(
            driver="rclone",
            driver_options={"vfs-cache-mode": "off"},
        ),
        _CustomModalCloudBucketStrategy(secret_name="named-modal-secret"),
    ],
)
def test_unknown_sdk_strategy_subclasses_cannot_retain_opaque_authority(
    strategy: MountStrategyBase,
) -> None:
    manifest = Manifest(entries={"data": S3Mount(bucket="bucket", mount_strategy=strategy)})

    with pytest.raises(MountConfigError, match="custom mount strategies"):
        validate_manifest_mount_credential_boundaries(manifest, provider_backend_id="docker")


def test_custom_strategy_rejects_pydantic_extra_configuration() -> None:
    class ExtraStrategy(DockerVolumeMountStrategy):
        type: Literal["extra_strategy"] = "extra_strategy"  # type: ignore[assignment]
        model_config = ConfigDict(extra="allow")

    strategy = ExtraStrategy.model_validate(
        {"driver": "rclone", "api_token": "custom-strategy-extra-secret"}
    )
    manifest = Manifest(entries={"data": S3Mount(bucket="bucket", mount_strategy=strategy)})

    with pytest.raises(MountConfigError, match="custom mount strategies") as exc:
        validate_manifest_mount_credential_boundaries(manifest)

    assert "custom-strategy-extra-secret" not in str(exc.value)


def test_custom_strategy_cannot_forge_builtin_class_provenance() -> None:
    original_class = MountStrategyBase._subclass_registry["in_container"]
    forged_class = type(
        "InContainerMountStrategy",
        (InContainerMountStrategy,),
        {
            "__module__": InContainerMountStrategy.__module__,
            "__qualname__": InContainerMountStrategy.__qualname__,
            "__annotations__": {"api_token": str | None},
            "api_token": None,
        },
    )
    try:
        strategy = cast(Any, forged_class)(
            pattern=RcloneMountPattern(),
            api_token="forged-strategy-secret",
        )
        manifest = Manifest(entries={"data": S3Mount(bucket="bucket", mount_strategy=strategy)})

        with pytest.raises(MountConfigError, match="custom mount strategies"):
            validate_manifest_mount_credential_boundaries(manifest)
    finally:
        MountStrategyBase._subclass_registry["in_container"] = original_class


def test_behavior_only_mount_strategy_is_rejected_before_activate() -> None:
    class BehaviorOnlyStrategy(InContainerMountStrategy):
        type: Literal["behavior_only_strategy"] = "behavior_only_strategy"  # type: ignore[assignment]
        activate_called: ClassVar[bool] = False

        async def activate(
            self,
            mount: Mount,
            session: Any,
            dest: Path,
            base_dir: Path,
        ) -> list[Any]:
            _ = (mount, session, dest, base_dir)
            type(self).activate_called = True
            return []

    strategy = BehaviorOnlyStrategy(pattern=RcloneMountPattern())
    manifest = Manifest(entries={"data": S3Mount(bucket="bucket", mount_strategy=strategy)})

    with pytest.raises(MountConfigError, match="custom mount strategies"):
        validate_manifest_mount_credential_boundaries(manifest)

    assert BehaviorOnlyStrategy.activate_called is False


@pytest.mark.asyncio
@pytest.mark.parametrize("credentialed", [False, True])
async def test_mount_apply_rejects_behavior_only_mount_strategy(
    credentialed: bool,
) -> None:
    sentinel = "direct-apply-secret"

    class BehaviorOnlyStrategy(InContainerMountStrategy):
        type: Literal["direct_apply_behavior_only"] = "direct_apply_behavior_only"  # type: ignore[assignment]
        activate_called: ClassVar[bool] = False

        async def activate(
            self,
            mount: Mount,
            session: Any,
            dest: Path,
            base_dir: Path,
        ) -> list[Any]:
            _ = (mount, session, dest, base_dir)
            type(self).activate_called = True
            return []

    mount = S3Mount(
        bucket="bucket",
        access_key_id="access-key" if credentialed else None,
        secret_access_key=sentinel if credentialed else None,
        mount_strategy=BehaviorOnlyStrategy(pattern=RcloneMountPattern()),
    )
    session = cast(Any, type("Session", (), {"state": type("State", (), {"type": "test"})()})())

    with pytest.raises(MountConfigError) as exc:
        await mount.apply(session, Path("/workspace/data"), Path("/workspace"))

    assert BehaviorOnlyStrategy.activate_called is False
    assert sentinel not in str(exc.value)


def test_custom_mount_pattern_fields_are_rejected_before_apply() -> None:
    class CustomRclonePattern(RcloneMountPattern):
        api_token: str | None = None
        apply_called: ClassVar[bool] = False

        async def apply(self, session: Any, path: Path, config: Any) -> None:
            _ = (session, path, config)
            type(self).apply_called = True

    pattern = CustomRclonePattern(api_token="custom-pattern-secret")
    manifest = Manifest(
        entries={
            "data": S3Mount(
                bucket="bucket",
                mount_strategy=InContainerMountStrategy(pattern=pattern),
            )
        }
    )

    with pytest.raises(MountConfigError, match="custom mount patterns") as exc:
        validate_manifest_mount_credential_boundaries(manifest)

    assert CustomRclonePattern.apply_called is False
    assert "custom-pattern-secret" not in str(exc.value)


def test_behavior_only_mount_pattern_is_rejected_before_apply() -> None:
    class BehaviorOnlyRclonePattern(RcloneMountPattern):
        apply_called: ClassVar[bool] = False

        async def apply(self, session: Any, path: Path, config: Any) -> None:
            _ = (session, path, config)
            type(self).apply_called = True

    pattern = BehaviorOnlyRclonePattern()
    manifest = Manifest(
        entries={
            "data": S3Mount(
                bucket="bucket",
                mount_strategy=InContainerMountStrategy(pattern=pattern),
            )
        }
    )

    with pytest.raises(MountConfigError, match="custom mount patterns"):
        validate_manifest_mount_credential_boundaries(manifest)

    assert BehaviorOnlyRclonePattern.apply_called is False


def test_mount_activation_rejects_custom_pattern_before_deepcopy() -> None:
    sentinel = "custom-pattern-deepcopy-secret"

    class CustomRclonePattern(RcloneMountPattern):
        deepcopy_called: ClassVar[bool] = False

        def __deepcopy__(self, memo: dict[int, Any] | None = None) -> CustomRclonePattern:
            _ = memo
            type(self).deepcopy_called = True
            raise RuntimeError(sentinel)

    strategy = InContainerMountStrategy(pattern=CustomRclonePattern())
    mount = S3Mount(bucket="bucket", mount_strategy=strategy)

    with pytest.raises(MountConfigError, match="custom mount patterns") as exc_info:
        validate_mount_activation_credential_boundary(mount, strategy)

    assert CustomRclonePattern.deepcopy_called is False
    assert sentinel not in repr(exc_info.value)


def test_ignores_environment_values_already_exposed_to_the_sandbox() -> None:
    manifest = Manifest(
        entries={
            "data": _s3_mount(strategy=InContainerMountStrategy(pattern=MountpointMountPattern()))
        },
        environment=Environment(
            value={"AWS_SECRET_ACCESS_KEY": "secret", "GITHUB_TOKEN": "unrelated"}
        ),
    )

    validate_manifest_mount_credential_boundaries(manifest)


def test_blobfuse_mounts_require_broad_acknowledgement() -> None:
    manifest = Manifest(
        entries={
            "data": AzureBlobMount(
                account="example",
                container="public",
                mount_strategy=InContainerMountStrategy(pattern=FuseMountPattern()),
            )
        }
    )

    with pytest.raises(MountConfigError, match="broad credential authority"):
        validate_manifest_mount_credential_boundaries(manifest)

    validate_manifest_mount_credential_boundaries(
        manifest.with_in_container_mount_broad_credential_exposure_acknowledged("data")
    )


def test_blobfuse_account_key_requires_mount_scoped_and_broad_acknowledgement() -> None:
    manifest = Manifest(
        entries={
            "data": AzureBlobMount(
                account="example",
                container="private",
                account_key="account-key",
                mount_strategy=InContainerMountStrategy(pattern=FuseMountPattern()),
            )
        }
    )

    mount_scoped = manifest.with_in_container_mount_credential_exposure_acknowledged("data")
    with pytest.raises(MountConfigError, match="broad credential authority"):
        validate_manifest_mount_credential_boundaries(mount_scoped)

    broad = manifest.with_in_container_mount_broad_credential_exposure_acknowledged("data")
    with pytest.raises(MountConfigError, match="mount-scoped credentials"):
        validate_manifest_mount_credential_boundaries(broad)

    validate_manifest_mount_credential_boundaries(
        mount_scoped.with_in_container_mount_broad_credential_exposure_acknowledged("data")
    )


def test_s3_files_require_broad_acknowledgement_before_ambient_iam_can_be_used() -> None:
    safe = Manifest(
        entries={
            "data": S3FilesMount(
                file_system_id="fs-123",
                extra_options={"tlsport": "4049"},
                mount_strategy=InContainerMountStrategy(pattern=S3FilesMountPattern()),
            )
        }
    )
    with pytest.raises(MountConfigError, match="broad credential authority"):
        validate_manifest_mount_credential_boundaries(safe)

    validate_manifest_mount_credential_boundaries(
        safe.with_in_container_mount_broad_credential_exposure_acknowledged("data")
    )


@pytest.mark.parametrize(
    "mount",
    [
        AzureBlobMount(
            account="example",
            container="public",
            mount_strategy=_CustomInContainerStrategy(pattern=FuseMountPattern()),
        ),
        S3FilesMount(
            file_system_id="fs-123",
            mount_strategy=_CustomInContainerStrategy(pattern=S3FilesMountPattern()),
        ),
    ],
)
def test_rejects_credential_required_patterns_in_inherited_in_container_strategies(
    mount: Any,
) -> None:
    with pytest.raises(MountConfigError, match="custom mount strategies"):
        validate_manifest_mount_credential_boundaries(Manifest(entries={"data": mount}))


def test_acknowledgement_requires_a_matching_provider_owned_strategy() -> None:
    strategy = InContainerMountStrategy(pattern=RcloneMountPattern())
    cast(Any, strategy).type = "vercel_cloud_bucket"
    manifest = Manifest(entries={"data": _s3_mount(strategy=strategy, credentialed=True)})

    with pytest.raises(MountConfigError, match="sandbox mount configuration is invalid"):
        manifest.with_in_container_mount_credential_exposure_acknowledged("data")


@pytest.mark.parametrize(
    "extra_args",
    [
        ["--config=/workspace/credentials.conf"],
        ["--s3-env-auth=true"],
        ["--s3-profile=production"],
        ["--azureblob-use-msi=true"],
        ["--header", "Authorization: Bearer secret"],
    ],
)
def test_rejects_rclone_credential_source_overrides(extra_args: list[str]) -> None:
    manifest = Manifest(
        entries={
            "data": _s3_mount(
                strategy=InContainerMountStrategy(
                    pattern=RcloneMountPattern(extra_args=extra_args),
                )
            )
        }
    )

    with pytest.raises(MountConfigError, match="does not support exposing"):
        validate_manifest_mount_credential_boundaries(manifest)


@pytest.mark.parametrize(
    ("strategy", "backend_id"),
    [
        (InContainerMountStrategy(pattern=RcloneMountPattern()), None),
        (DaytonaCloudBucketMountStrategy(), "daytona"),
    ],
)
def test_box_mounts_with_direct_credentials_require_exact_acknowledgement(
    strategy: MountStrategyBase,
    backend_id: str | None,
) -> None:
    manifest = Manifest(
        entries={
            "data": BoxMount(
                access_token="box-access-token",
                mount_strategy=strategy,
            )
        }
    )

    with pytest.raises(MountConfigError, match="mount-scoped credentials"):
        validate_manifest_mount_credential_boundaries(
            manifest,
            provider_backend_id=backend_id,
        )

    validate_manifest_mount_credential_boundaries(
        manifest.with_in_container_mount_credential_exposure_acknowledged("data"),
        provider_backend_id=backend_id,
    )


def test_box_config_file_requires_broad_acknowledgement() -> None:
    manifest = Manifest(
        entries={
            "data": BoxMount(
                box_config_file="/run/secrets/box.json",
                mount_strategy=InContainerMountStrategy(pattern=RcloneMountPattern()),
            )
        }
    )

    with pytest.raises(MountConfigError, match="broad credential authority"):
        validate_manifest_mount_credential_boundaries(manifest)
    with pytest.raises(MountConfigError, match="broad credential authority"):
        validate_manifest_mount_credential_boundaries(
            manifest.with_in_container_mount_credential_exposure_acknowledged("data")
        )
    validate_manifest_mount_credential_boundaries(
        manifest.with_in_container_mount_broad_credential_exposure_acknowledged("data")
    )


@pytest.mark.parametrize(
    "mount",
    [
        BoxMount(mount_strategy=InContainerMountStrategy(pattern=RcloneMountPattern())),
        BoxMount(
            client_id="client-id",
            mount_strategy=InContainerMountStrategy(pattern=RcloneMountPattern()),
        ),
        BoxMount(
            client_secret="client-secret",
            mount_strategy=InContainerMountStrategy(pattern=RcloneMountPattern()),
        ),
    ],
)
def test_box_in_container_mount_requires_non_interactive_authentication(
    mount: BoxMount,
) -> None:
    manifest = Manifest(entries={"data": mount})

    with pytest.raises(MountConfigError, match="non-interactive authentication source"):
        validate_manifest_mount_credential_boundaries(manifest)


@pytest.mark.parametrize(
    ("mount", "broad"),
    [
        (
            BoxMount(
                access_token=value,
                mount_strategy=InContainerMountStrategy(pattern=RcloneMountPattern()),
            ),
            False,
        )
        for value in ("", "   ")
    ]
    + [
        (
            BoxMount(
                token=value,
                mount_strategy=InContainerMountStrategy(pattern=RcloneMountPattern()),
            ),
            False,
        )
        for value in ("", "   ")
    ]
    + [
        (
            BoxMount(
                config_credentials=value,
                mount_strategy=InContainerMountStrategy(pattern=RcloneMountPattern()),
            ),
            False,
        )
        for value in ("", "   ")
    ]
    + [
        (
            BoxMount(
                box_config_file=value,
                mount_strategy=InContainerMountStrategy(pattern=RcloneMountPattern()),
            ),
            True,
        )
        for value in ("", "   ")
    ],
)
def test_box_in_container_mount_rejects_empty_authentication_sources(
    mount: BoxMount,
    broad: bool,
) -> None:
    manifest = Manifest(entries={"data": mount})
    acknowledged = (
        manifest.with_in_container_mount_broad_credential_exposure_acknowledged("data")
        if broad
        else manifest.with_in_container_mount_credential_exposure_acknowledged("data")
    )

    with pytest.raises(MountConfigError, match="authentication values must not be empty"):
        validate_manifest_mount_credential_boundaries(acknowledged)


@pytest.mark.parametrize(
    ("mount", "broad", "invalid_field"),
    [
        (
            BoxMount(
                access_token="box-access-token",
                box_config_file=value,
                mount_strategy=InContainerMountStrategy(pattern=RcloneMountPattern()),
            ),
            False,
            "box_config_file",
        )
        for value in ("", "   ")
    ]
    + [
        (
            BoxMount(
                access_token=value,
                box_config_file="/run/secrets/box.json",
                mount_strategy=InContainerMountStrategy(pattern=RcloneMountPattern()),
            ),
            True,
            "access_token",
        )
        for value in ("", "   ")
    ],
)
def test_box_in_container_mount_rejects_mixed_usable_and_empty_authentication_sources(
    mount: BoxMount,
    broad: bool,
    invalid_field: str,
) -> None:
    manifest = Manifest(entries={"data": mount})
    acknowledged = (
        manifest.with_in_container_mount_broad_credential_exposure_acknowledged("data")
        if broad
        else manifest.with_in_container_mount_credential_exposure_acknowledged("data")
    )

    with pytest.raises(MountConfigError, match="authentication values must not be empty") as exc:
        validate_manifest_mount_credential_boundaries(acknowledged)

    assert exc.value.context["credential_fields"] == (invalid_field,)


def test_preserves_box_mounts_with_an_external_strategy() -> None:
    manifest = Manifest(
        entries={
            "data": BoxMount(
                access_token="box-access-token",
                mount_strategy=DockerVolumeMountStrategy(driver="rclone"),
            )
        }
    )

    validate_manifest_mount_credential_boundaries(manifest, provider_backend_id="docker")


def test_preserves_multiline_external_mount_credentials() -> None:
    manifest = Manifest(
        entries={
            "data": GCSMount(
                bucket="bucket",
                service_account_credentials='{"private_key":"line-1\nline-2"}',
                mount_strategy=DockerVolumeMountStrategy(driver="rclone"),
            )
        }
    )

    validate_manifest_mount_credential_boundaries(manifest, provider_backend_id="docker")


@pytest.mark.parametrize(
    ("module_name", "environment"),
    [
        (
            "examples.sandbox.docker.mounts.azure_mount_read_write",
            {
                "AZURE_STORAGE_ACCOUNT": "account",
                "AZURE_STORAGE_CONTAINER": "container",
                "AZURE_STORAGE_ACCOUNT_KEY": "example-key",
            },
        ),
        (
            "examples.sandbox.docker.mounts.gcs_mount_read_write",
            {
                "GCS_MOUNT_BUCKET": "bucket",
                "GCS_ACCESS_ID": "example-access-id",
                "GCS_SECRET_ACCESS_KEY": "example-secret-key",
            },
        ),
    ],
)
def test_docker_mount_examples_use_supported_external_strategies(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    environment: dict[str, str],
) -> None:
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    module = importlib.import_module(module_name)

    cases = module._mount_cases()

    assert [case.name for case in cases] == ["docker_volume/rclone"]
    for case in cases:
        assert isinstance(case.mount.mount_strategy, DockerVolumeMountStrategy)
        validate_manifest_mount_credential_boundaries(
            Manifest(entries={case.mount_dir: case.mount}),
            provider_backend_id="docker",
        )


@pytest.mark.parametrize(
    ("mount", "field_name"),
    [
        (
            S3Mount(
                bucket="bucket",
                s3_provider="AWS\naccess_key_id = injected-value",
                mount_strategy=InContainerMountStrategy(pattern=RcloneMountPattern()),
            ),
            "s3_provider",
        ),
        (
            AzureBlobMount(
                account="account\nkey = injected-value",
                container="container",
                mount_strategy=InContainerMountStrategy(pattern=RcloneMountPattern()),
            ),
            "account",
        ),
        (
            R2Mount(
                bucket="bucket",
                account_id="account\nsecret_access_key = injected-value",
                mount_strategy=InContainerMountStrategy(pattern=RcloneMountPattern()),
            ),
            "account_id",
        ),
    ],
)
def test_rejects_and_redacts_rclone_config_line_injection(
    mount: S3Mount | AzureBlobMount | R2Mount,
    field_name: str,
) -> None:
    manifest = Manifest(entries={"data": mount})

    with pytest.raises(MountConfigError, match="must not contain line breaks") as exc:
        validate_manifest_mount_credential_boundaries(manifest)

    assert exc.value.context["configuration_fields"] == (field_name,)
    assert "injected-value" not in str(exc.value)

    sanitized, redacted = sanitize_manifest_mount_authority(manifest)
    sanitized_mount = sanitized.entries["data"]
    assert redacted is True
    assert getattr(sanitized_mount, field_name) == ""
    assert "injected-value" not in repr(sanitized)


@pytest.mark.parametrize(
    ("mount", "field_name"),
    [
        (
            S3Mount(
                bucket="bucket",
                endpoint_url=("https://s3.example,public_bucket=0,passwd_file=/workspace/creds"),
                mount_strategy=BlaxelCloudBucketMountStrategy(),
            ),
            "endpoint_url",
        ),
        (
            S3Mount(
                bucket="bucket",
                region="us-east-1,public_bucket=0,passwd_file=/workspace/creds",
                mount_strategy=BlaxelCloudBucketMountStrategy(),
            ),
            "region",
        ),
        (
            R2Mount(
                bucket="bucket",
                account_id="account",
                custom_domain=("https://r2.example,public_bucket=0,passwd_file=/workspace/creds"),
                mount_strategy=BlaxelCloudBucketMountStrategy(),
            ),
            "custom_domain",
        ),
        (
            R2Mount(
                bucket="bucket",
                account_id="account,public_bucket=0,passwd_file=/workspace/creds",
                mount_strategy=BlaxelCloudBucketMountStrategy(),
            ),
            "account_id",
        ),
    ],
)
def test_rejects_blaxel_s3fs_endpoint_option_injection(
    mount: S3Mount | R2Mount,
    field_name: str,
) -> None:
    sentinel = "s3fs-endpoint-secret"
    manifest = Manifest(
        entries={
            "creds": File(content=sentinel.encode()),
            "data": mount,
        }
    )

    with pytest.raises(MountConfigError, match="must not contain s3fs option delimiters") as exc:
        validate_manifest_mount_credential_boundaries(
            manifest,
            provider_backend_id="blaxel",
        )

    assert exc.value.context["configuration_fields"] == (field_name,)
    assert sentinel not in str(exc.value)

    state = TestSessionState(manifest=manifest, snapshot=NoopSnapshot(id="snapshot"))
    with pytest.raises(MountConfigError) as serialization_exc:
        _SecurityTestClient().serialize_session_state(state)
    assert sentinel not in str(serialization_exc.value)


def test_rejects_rclone_on_the_fly_remote_name() -> None:
    sentinel = "remote-name-secret"
    manifest = Manifest(
        entries={
            "data": _s3_mount(
                strategy=InContainerMountStrategy(
                    pattern=RcloneMountPattern(
                        remote_name=f":s3,access_key_id=access,secret_access_key={sentinel}"
                    )
                )
            )
        }
    )

    with pytest.raises(MountConfigError, match="does not support exposing") as exc:
        validate_manifest_mount_credential_boundaries(manifest)

    assert sentinel not in str(exc.value)


def test_serialization_redacts_rclone_on_the_fly_remote_name() -> None:
    sentinel = "serialized-remote-name-secret"
    state = TestSessionState(
        manifest=Manifest(
            entries={
                "data": _s3_mount(
                    strategy=InContainerMountStrategy(
                        pattern=RcloneMountPattern(remote_name=f":s3,secret_access_key={sentinel}")
                    )
                )
            }
        ),
        snapshot=NoopSnapshot(id="snapshot"),
    )

    payload = _SecurityTestClient().serialize_session_state(state)

    pattern = payload["manifest"]["entries"]["data"]["mount_strategy"]["pattern"]  # type: ignore[index]
    assert pattern["remote_name"] is None
    assert payload[REDACTED_MOUNT_AUTHORITY_KEY] is True
    assert sentinel not in repr(payload)


def test_preserves_ordinary_rclone_remote_name() -> None:
    manifest = Manifest(
        entries={
            "data": _s3_mount(
                strategy=InContainerMountStrategy(
                    pattern=RcloneMountPattern(remote_name="public bucket-1")
                )
            )
        }
    )

    validate_manifest_mount_credential_boundaries(manifest)
    sanitized, redacted = sanitize_raw_session_state_mount_authority(
        {
            "type": "test",
            "manifest": manifest.model_dump(mode="json"),
            "snapshot": NoopSnapshot(id="snapshot").model_dump(mode="json"),
        }
    )

    assert redacted is False
    assert (
        sanitized["manifest"]["entries"]["data"]["mount_strategy"]["pattern"][  # type: ignore[index]
            "remote_name"
        ]
        == "public bucket-1"
    )


def test_preserves_supported_credentialless_rclone_extra_args() -> None:
    manifest = Manifest(
        entries={
            "data": _s3_mount(
                strategy=InContainerMountStrategy(
                    pattern=RcloneMountPattern(
                        extra_args=[
                            "--allow-other",
                            "--uid",
                            "123",
                            "--gid=456",
                            "--buffer-size",
                            "0",
                        ]
                    ),
                )
            )
        }
    )

    validate_manifest_mount_credential_boundaries(manifest)


@pytest.mark.parametrize(
    "endpoint_url",
    [
        "https://user:malformed-secret@[invalid",
        "https:user:malformed-secret@example.test",
    ],
)
def test_rejects_malformed_inline_credential_url_without_mutating_trusted_manifest(
    endpoint_url: str,
) -> None:
    manifest = Manifest(
        entries={
            "data": S3Mount(
                bucket="example-bucket",
                endpoint_url=endpoint_url,
                mount_strategy=InContainerMountStrategy(pattern=RcloneMountPattern()),
            )
        }
    )

    with pytest.raises(MountConfigError, match="does not support exposing"):
        validate_manifest_mount_credential_boundaries(manifest)

    mount = manifest.entries["data"]
    assert isinstance(mount, S3Mount)
    assert mount.endpoint_url == endpoint_url


@pytest.mark.parametrize(
    "endpoint_url",
    [
        "https://user:pattern-secret@example.test",
        "https://example.test?signature=pattern-secret",
    ],
)
def test_rejects_mountpoint_endpoint_authority(endpoint_url: str) -> None:
    manifest = Manifest(
        entries={
            "data": _s3_mount(
                strategy=InContainerMountStrategy(
                    pattern=MountpointMountPattern(
                        options=MountpointMountPattern.MountpointOptions(
                            endpoint_url=endpoint_url,
                        )
                    )
                )
            )
        }
    )

    with pytest.raises(MountConfigError, match="does not support exposing") as exc:
        validate_manifest_mount_credential_boundaries(manifest)

    assert exc.value.context["credential_fields"] == (
        "mount_strategy.pattern.options.endpoint_url",
    )
    assert "pattern-secret" not in str(exc.value)


@pytest.mark.parametrize(
    ("mount", "credential_path"),
    [
        (
            GCSMount(
                bucket="bucket",
                service_account_file="/workspace/credentials.json",
                mount_strategy=DockerVolumeMountStrategy(driver="rclone"),
            ),
            "/workspace/credentials.json",
        ),
        (
            BoxMount(
                box_config_file="credentials.json",
                mount_strategy=DockerVolumeMountStrategy(driver="rclone"),
            ),
            "credentials.json",
        ),
    ],
)
def test_rejects_manifest_backed_credential_files(
    mount: GCSMount | BoxMount,
    credential_path: str,
) -> None:
    _ = credential_path
    manifest = Manifest(
        entries={
            "credentials.json": File(content=b"credential-file-secret"),
            "data": mount,
        }
    )

    with pytest.raises(MountConfigError, match="credential files stored in the manifest"):
        validate_manifest_mount_credential_boundaries(
            manifest,
            provider_backend_id="docker",
        )


def test_broad_acknowledgement_does_not_allow_manifest_backed_rclone_config() -> None:
    manifest = Manifest(
        entries={
            "credentials.conf": File(content=b"credential-file-secret"),
            "data": S3Mount(
                bucket="bucket",
                mount_strategy=InContainerMountStrategy(
                    pattern=RcloneMountPattern(config_file_path=Path("credentials.conf"))
                ),
            ),
        }
    ).with_in_container_mount_broad_credential_exposure_acknowledged("data")

    with pytest.raises(MountConfigError, match="credential files stored in the manifest"):
        validate_manifest_mount_credential_boundaries(manifest)


@pytest.mark.parametrize(
    ("credential_path", "source"),
    [
        ("/workspace/credentials.json", LocalFile(src=Path("credentials.json"))),
        ("/workspace/imported/credentials.json", LocalDir(src=Path("imported"))),
        (
            "/workspace/repository/credentials.json",
            GitRepo(repo="example/repository", ref="main"),
        ),
        (
            "/workspace/secrets/credentials.json",
            S3Mount(
                bucket="secret-bucket",
                mount_strategy=DockerVolumeMountStrategy(driver="rclone"),
            ),
        ),
    ],
)
def test_rejects_credential_files_from_manifest_materialization_sources(
    credential_path: str,
    source: BaseEntry,
) -> None:
    source_path = credential_path.removeprefix("/workspace/").split("/", 1)[0]
    if credential_path == "/workspace/credentials.json":
        source_path = "credentials.json"
    manifest = Manifest(
        entries={
            source_path: source,
            "data": GCSMount(
                bucket="bucket",
                service_account_file=credential_path,
                mount_strategy=DockerVolumeMountStrategy(driver="rclone"),
            ),
        }
    )

    with pytest.raises(MountConfigError, match="credential files stored in the manifest"):
        validate_manifest_mount_credential_boundaries(
            manifest,
            provider_backend_id="docker",
        )


def test_session_state_serialization_redacts_complete_opaque_authority_fields() -> None:
    manifest = Manifest(
        entries={
            "data": _s3_mount(
                strategy=DockerVolumeMountStrategy(
                    driver="rclone",
                    driver_options={
                        "vfs-cache-mode": "off",
                        "s3-secret-access-key": "driver-secret",
                        "s3-env-auth": "true",
                        "config": "/host/rclone.conf",
                    },
                ),
                credentialed=True,
            )
        }
    )
    state = TestSessionState(
        manifest=manifest,
        snapshot=NoopSnapshot(id="snapshot"),
    )
    client = _SecurityTestClient()

    payload = client.serialize_session_state(state)
    serialized_mount = payload["manifest"]["entries"]["data"]  # type: ignore[index]

    assert payload[REDACTED_MOUNT_AUTHORITY_KEY] is True
    assert serialized_mount["access_key_id"] is None
    assert serialized_mount["secret_access_key"] is None
    assert serialized_mount["mount_strategy"]["driver_options"] == {}
    assert "example-secret-key" not in repr(payload)
    assert "driver-secret" not in repr(payload)

    restored = client.deserialize_session_state(payload)
    assert restored.mount_authority_redacted is True

    trusted_manifest = manifest.model_copy(deep=True)
    rebound = restored.rebind_persisted_mount_authority(
        trusted_manifest,
        provider_backend_id="docker",
    )
    rebound_mount = rebound.manifest.entries["data"]
    assert isinstance(rebound_mount, S3Mount)
    trusted_mount = trusted_manifest.entries["data"]
    assert isinstance(trusted_mount, S3Mount)
    assert rebound_mount.access_key_id == "example-access-key"
    assert rebound_mount.secret_access_key == "example-secret-key"
    assert rebound_mount.mount_strategy == trusted_mount.mount_strategy
    assert rebound.mount_authority_redacted is False
    assert rebound.mount_authority_rebound is True
    validate_manifest_mount_credential_boundaries(
        rebound.manifest,
        provider_backend_id="docker",
    )


def test_session_state_round_trip_preserves_credentialless_external_mount() -> None:
    manifest = Manifest(
        entries={
            "data": S3Mount(
                bucket="example-bucket",
                mount_strategy=DockerVolumeMountStrategy(driver="rclone"),
            )
        }
    )
    state = TestSessionState(manifest=manifest, snapshot=NoopSnapshot(id="snapshot"))
    client = _SecurityTestClient()

    payload = client.serialize_session_state(state)
    restored = client.deserialize_session_state(payload)

    assert CREDENTIALLESS_MOUNT_AUTHORITY_KEY not in payload
    assert REDACTED_MOUNT_AUTHORITY_KEY not in payload
    assert restored.manifest == manifest
    assert restored.mount_authority_redacted is False
    assert restored.mount_authority_rebound is False


def test_credentialless_marker_does_not_override_configured_mount_authority() -> None:
    sentinel = "configured-secret-access-key"
    payload: dict[str, object] = {
        "type": "test",
        "manifest": Manifest(
            entries={
                "data": S3Mount(
                    bucket="example-bucket",
                    access_key_id="access-key",
                    secret_access_key=sentinel,
                    mount_strategy=DockerVolumeMountStrategy(driver="rclone"),
                )
            }
        ).model_dump(mode="json"),
        "snapshot": NoopSnapshot(id="snapshot").model_dump(mode="json"),
        CREDENTIALLESS_MOUNT_AUTHORITY_KEY: True,
    }

    restored = _SecurityTestClient().deserialize_session_state(payload)

    assert restored.mount_authority_redacted is True
    assert CREDENTIALLESS_MOUNT_AUTHORITY_KEY not in payload
    assert payload[REDACTED_MOUNT_AUTHORITY_KEY] is True
    assert sentinel not in repr(payload)


def test_session_state_serialization_preserves_custom_non_mount_fields() -> None:
    state = TestSessionState(
        manifest=Manifest(entries={"custom": _CustomTokenEntry(token="ordinary-token-value")}),
        snapshot=NoopSnapshot(id="snapshot"),
    )

    payload = _SecurityTestClient().serialize_session_state(state)
    restored = _SecurityTestClient().deserialize_session_state(payload)

    assert REDACTED_MOUNT_AUTHORITY_KEY not in payload
    assert payload["manifest"]["entries"]["custom"]["token"] == "ordinary-token-value"  # type: ignore[index]
    entry = restored.manifest.entries["custom"]
    assert isinstance(entry, _CustomTokenEntry)
    assert entry.token == "ordinary-token-value"


@pytest.mark.parametrize(
    "children",
    [
        "ordinary-metadata",
        {
            "nested": {
                "type": "s3_mount",
                "access_key_id": "ordinary-access-metadata",
                "secret_access_key": "ordinary-secret-metadata",
            }
        },
    ],
)
def test_session_state_serialization_preserves_custom_non_dir_children(children: Any) -> None:
    state = TestSessionState(
        manifest=Manifest(entries={"custom": _CustomChildrenEntry(children=children)}),
        snapshot=NoopSnapshot(id="snapshot"),
    )

    payload = _SecurityTestClient().serialize_session_state(state)
    restored = _SecurityTestClient().deserialize_session_state(payload)

    assert REDACTED_MOUNT_AUTHORITY_KEY not in payload
    assert payload["manifest"]["entries"]["custom"]["children"] == children  # type: ignore[index]
    entry = restored.manifest.entries["custom"]
    assert isinstance(entry, _CustomChildrenEntry)
    assert entry.children == children


def test_session_state_serialization_rejects_registered_custom_strategy_configuration() -> None:
    pattern = {
        "type": "custom_pattern",
        "extra_args": ["--ordinary-option"],
        "remote_name": "ordinary-remote",
        "options": {"endpoint_url": "https://public.example.test"},
    }
    state = TestSessionState(
        manifest=Manifest(
            entries={
                "data": S3Mount(
                    bucket="example-bucket",
                    mount_strategy=_CustomPatternStrategy(
                        pattern=pattern,
                        api_token="custom-strategy-secret",
                    ),
                )
            }
        ),
        snapshot=NoopSnapshot(id="snapshot"),
    )

    with pytest.raises(MountConfigError) as exc:
        _SecurityTestClient().serialize_session_state(state)

    assert "custom-strategy-secret" not in str(exc.value)


def test_session_state_serialization_redacts_custom_strategy_with_known_discriminator() -> None:
    strategy = _CustomPatternStrategy(
        pattern={"type": "custom_pattern"},
        api_token="custom-strategy-secret",
    )
    cast(Any, strategy).type = "docker_volume"
    state = TestSessionState(
        manifest=Manifest(
            entries={
                "data": S3Mount(
                    bucket="example-bucket",
                    mount_strategy=strategy,
                )
            }
        ),
        snapshot=NoopSnapshot(id="snapshot"),
    )

    with pytest.raises(MountConfigError) as exc_info:
        _SecurityTestClient().serialize_session_state(state)

    assert "custom-strategy-secret" not in repr(exc_info.value)


def test_rejects_configured_custom_mount_strategies_before_side_effects() -> None:
    manifest = Manifest(
        entries={
            "data": S3Mount(
                bucket="example-bucket",
                mount_strategy=_CustomPatternStrategy(
                    pattern={},
                    api_token="custom-strategy-secret",
                ),
            )
        }
    )

    with pytest.raises(MountConfigError, match="custom mount strategies") as exc:
        validate_manifest_mount_credential_boundaries(manifest)

    assert "custom-strategy-secret" not in str(exc.value)


def test_custom_entry_at_credential_file_path_is_rejected() -> None:
    manifest = Manifest(
        entries={
            "credentials.json": _CustomTokenEntry(token="custom-source"),
            "data": GCSMount(
                bucket="bucket",
                service_account_file="/workspace/credentials.json",
                mount_strategy=DockerVolumeMountStrategy(driver="rclone"),
            ),
        }
    )

    with pytest.raises(MountConfigError, match="credential files stored in the manifest"):
        validate_manifest_mount_credential_boundaries(
            manifest,
            provider_backend_id="docker",
        )


def test_session_state_serialization_rejects_custom_credential_file_materializer() -> None:
    sentinel = "custom-source-secondary-secret"
    state = TestSessionState(
        manifest=Manifest(
            entries={
                "credentials.json": _CustomCredentialSourceEntry(
                    content="ordinary-content",
                    source_token=sentinel,
                ),
                "data": GCSMount(
                    bucket="bucket",
                    service_account_file="/workspace/credentials.json",
                    mount_strategy=DockerVolumeMountStrategy(driver="rclone"),
                ),
            }
        ),
        snapshot=NoopSnapshot(id="snapshot"),
    )

    with pytest.raises(MountConfigError) as exc:
        _SecurityTestClient().serialize_session_state(state)

    assert sentinel not in str(exc.value)
    traceback = exc.value.__traceback__
    while traceback is not None:
        module_name = traceback.tb_frame.f_globals.get("__name__", "")
        if isinstance(module_name, str) and module_name.startswith("agents."):
            assert sentinel not in repr(traceback.tb_frame.f_locals)
        traceback = traceback.tb_next


def test_structural_local_dir_credential_path_remains_serializable() -> None:
    sentinel = "structural-local-dir-secret"
    state = TestSessionState(
        manifest=Manifest(
            entries={
                "credentials": LocalDir(src=None),
                "data": GCSMount(
                    bucket="bucket",
                    service_account_file="/workspace/credentials/key.json",
                    service_account_credentials=sentinel,
                    mount_strategy=DockerVolumeMountStrategy(driver="rclone"),
                ),
            }
        ),
        snapshot=NoopSnapshot(id="snapshot"),
    )

    validate_manifest_mount_credential_boundaries(state.manifest, provider_backend_id="docker")
    payload = _SecurityTestClient().serialize_session_state(state)

    assert payload[REDACTED_MOUNT_AUTHORITY_KEY] is True
    assert sentinel not in repr(payload)


@pytest.mark.parametrize(
    "backend_id",
    ["docker", "modal"],
)
def test_opaque_external_authority_remains_resumable_through_trusted_rebind(
    backend_id: str,
) -> None:
    if backend_id == "docker":
        strategy: MountStrategyBase = DockerVolumeMountStrategy(
            driver="rclone",
            driver_options={"vfs-cache-mode": "off"},
        )
    else:
        modal_mounts = importlib.import_module("agents.extensions.sandbox.modal.mounts")
        strategy = modal_mounts.ModalCloudBucketMountStrategy(
            secret_name="named-modal-secret",
            secret_environment_name="staging",
        )
    manifest = Manifest(
        entries={
            "data": S3Mount(
                bucket="example-bucket",
                mount_strategy=strategy,
            )
        }
    )
    client = _SecurityTestClient()
    state = TestSessionState(manifest=manifest, snapshot=NoopSnapshot(id="snapshot"))

    payload = client.serialize_session_state(state)
    restored = client.deserialize_session_state(payload)
    rebound = restored.rebind_persisted_mount_authority(
        manifest,
        provider_backend_id=backend_id,
    )

    assert payload[REDACTED_MOUNT_AUTHORITY_KEY] is True
    assert rebound.manifest == manifest
    assert rebound.mount_authority_redacted is False


def test_in_container_acknowledgement_is_rebound_only_from_trusted_manifest() -> None:
    manifest = Manifest(
        entries={
            "data": _s3_mount(
                strategy=InContainerMountStrategy(pattern=RcloneMountPattern()),
                credentialed=True,
            )
        }
    ).with_in_container_mount_credential_exposure_acknowledged("data")
    client = _SecurityTestClient()
    state = TestSessionState(manifest=manifest, snapshot=NoopSnapshot(id="snapshot"))

    payload = client.serialize_session_state(state)
    restored = client.deserialize_session_state(payload)

    assert "credential_exposure" not in repr(payload)
    with pytest.raises(ValueError, match="cannot be resumed"):
        restored.assert_path_grants_rebound()

    rebound = restored.rebind_persisted_mount_authority(
        manifest,
        provider_backend_id="docker",
    )
    validate_manifest_mount_credential_boundaries(
        rebound.manifest,
        provider_backend_id="docker",
    )
    assert rebound.manifest._acknowledges_in_container_mount_credential_exposure(
        "/workspace/data",
        "mount_scoped",
    )


@pytest.mark.parametrize(
    "mount",
    [
        AzureBlobMount(
            account="example",
            container="private",
            mount_strategy=InContainerMountStrategy(pattern=FuseMountPattern()),
        ),
        S3FilesMount(
            file_system_id="fs-123",
            mount_strategy=InContainerMountStrategy(pattern=S3FilesMountPattern()),
        ),
    ],
)
def test_implicit_broad_authority_is_rebound_only_from_trusted_manifest(
    mount: Mount,
) -> None:
    manifest = Manifest(
        entries={"data": mount}
    ).with_in_container_mount_broad_credential_exposure_acknowledged("data")
    client = _SecurityTestClient()
    state = TestSessionState(manifest=manifest, snapshot=NoopSnapshot(id="snapshot"))

    payload = client.serialize_session_state(state)
    restored = client.deserialize_session_state(payload)

    assert payload[REDACTED_MOUNT_AUTHORITY_KEY] is True
    assert "credential_exposure" not in repr(payload)
    assert restored.mount_authority_redacted is True
    rebound = restored.rebind_persisted_mount_authority(
        manifest,
        provider_backend_id="docker",
    )
    validate_manifest_mount_credential_boundaries(
        rebound.manifest,
        provider_backend_id="docker",
    )
    assert rebound.manifest._acknowledges_in_container_mount_credential_exposure(
        "/workspace/data",
        "broad",
    )


def test_mount_authority_rebind_requires_exact_credential_free_topology() -> None:
    original = Manifest(
        entries={
            "data": _s3_mount(
                strategy=DockerVolumeMountStrategy(driver="rclone"),
                credentialed=True,
            )
        }
    )
    state = TestSessionState(manifest=original, snapshot=NoopSnapshot(id="snapshot"))
    client = _SecurityTestClient()
    restored = client.deserialize_session_state(client.serialize_session_state(state))
    mismatched = original.model_copy(deep=True)
    mount = mismatched.entries["data"]
    assert isinstance(mount, S3Mount)
    mount.bucket = "different-bucket"

    with pytest.raises(MountConfigError):
        restored.rebind_persisted_mount_authority(
            mismatched,
            provider_backend_id="docker",
        )

    trusted_mount = mismatched.entries["data"]
    assert isinstance(trusted_mount, S3Mount)
    assert trusted_mount.access_key_id == "example-access-key"
    assert trusted_mount.secret_access_key == "example-secret-key"

    root_mismatched = original.model_copy(deep=True)
    root_mismatched.root = "/different-workspace"

    with pytest.raises(MountConfigError):
        restored.rebind_persisted_mount_authority(
            root_mismatched,
            provider_backend_id="docker",
        )


def test_resume_validation_rejects_wrong_provider_strategy() -> None:
    state = TestSessionState(
        manifest=Manifest(
            entries={
                "data": S3Mount(
                    bucket="example-bucket",
                    mount_strategy=DaytonaCloudBucketMountStrategy(),
                )
            }
        ),
        snapshot=NoopSnapshot(id="snapshot"),
    )

    with pytest.raises(MountConfigError, match="not supported by this sandbox backend"):
        state.assert_path_grants_rebound()


def test_session_state_serialization_redacts_pattern_authority() -> None:
    manifest = Manifest(
        entries={
            "credentials.conf": File(content=b"credential-file-secret"),
            "rclone": _s3_mount(
                strategy=InContainerMountStrategy(
                    pattern=RcloneMountPattern(
                        extra_args=[
                            "--vfs-cache-mode=off",
                            "--config=/workspace/credentials.conf",
                        ]
                    )
                )
            ),
            "s3files": S3FilesMount(
                file_system_id="fs-123",
                mount_strategy=InContainerMountStrategy(
                    pattern=S3FilesMountPattern(
                        options=S3FilesMountPattern.S3FilesOptions(
                            extra_options={
                                "tlsport": "4049",
                                "secret_access_key": "pattern-secret",
                            }
                        )
                    )
                ),
            ),
            "mountpoint": _s3_mount(
                strategy=InContainerMountStrategy(
                    pattern=MountpointMountPattern(
                        options=MountpointMountPattern.MountpointOptions(
                            endpoint_url="https://example.test?signature=pattern-secret"
                        )
                    )
                )
            ),
        }
    )
    state = TestSessionState(manifest=manifest, snapshot=NoopSnapshot(id="snapshot"))

    payload = _SecurityTestClient().serialize_session_state(state)
    entries = payload["manifest"]["entries"]  # type: ignore[index]

    assert payload[REDACTED_MOUNT_AUTHORITY_KEY] is True
    assert entries["credentials.conf"]["content"] == ""
    assert entries["rclone"]["mount_strategy"]["pattern"]["extra_args"] == []
    assert entries["s3files"]["mount_strategy"]["pattern"]["options"]["extra_options"] == {}
    assert entries["mountpoint"]["mount_strategy"]["pattern"]["options"]["endpoint_url"] is None
    assert "credential-file-secret" not in repr(payload)
    assert "pattern-secret" not in repr(payload)


def test_session_state_rejects_inherited_in_container_strategy() -> None:
    manifest = Manifest(
        entries={
            "credentials.conf": File(content=b"credential-file-secret"),
            "data": _s3_mount(
                strategy=_CustomInContainerStrategy(
                    pattern=RcloneMountPattern(
                        extra_args=["--config=/workspace/credentials.conf"],
                    )
                )
            ),
        }
    )

    with pytest.raises(MountConfigError) as exc:
        _SecurityTestClient().serialize_session_state(
            TestSessionState(manifest=manifest, snapshot=NoopSnapshot(id="snapshot"))
        )

    assert "credential-file-secret" not in str(exc.value)


def test_raw_state_sanitization_preserves_explicit_sandbox_environment() -> None:
    manifest = Manifest(
        entries={
            "credentials.json": File(content=b"credential-file-secret"),
            "data": GCSMount(
                bucket="bucket",
                service_account_file="/workspace/credentials.json",
                mount_strategy=InContainerMountStrategy(pattern=MountpointMountPattern()),
            ),
        }
    )
    payload: dict[str, object] = {
        "type": "test",
        "manifest": manifest.model_dump(mode="json"),
        "snapshot": NoopSnapshot(id="snapshot").model_dump(mode="json"),
        "base_envs": {
            "AWS_SECRET_ACCESS_KEY": "ambient-secret",
            "GITHUB_TOKEN": "unrelated",
        },
    }

    sanitized, redacted = sanitize_raw_session_state_mount_authority(payload)

    assert redacted is True
    assert isinstance(sanitized, dict)
    assert sanitized[REDACTED_MOUNT_AUTHORITY_KEY] is True
    assert sanitized["manifest"]["entries"]["credentials.json"]["content"] == ""
    assert sanitized["base_envs"] == {
        "AWS_SECRET_ACCESS_KEY": "ambient-secret",
        "GITHUB_TOKEN": "unrelated",
    }
    assert "credential-file-secret" not in repr(sanitized)
    assert "ambient-secret" in repr(sanitized)


def test_raw_state_sanitization_rejects_credential_content_without_file_discriminator() -> None:
    manifest = Manifest(
        entries={
            "credentials.json": File(content=b"credential-file-secret"),
            "data": GCSMount(
                bucket="bucket",
                service_account_file="/workspace/credentials.json",
                mount_strategy=DockerVolumeMountStrategy(driver="rclone"),
            ),
        }
    ).model_dump(mode="json")
    manifest["entries"]["credentials.json"]["type"] = "unknown_file"
    payload: dict[str, object] = {"manifest": manifest}

    with pytest.raises(ValueError) as exc:
        sanitize_raw_session_state_mount_authority(payload)

    assert "credential-file-secret" not in str(exc.value)


def test_legacy_non_inline_credential_file_source_cannot_survive_deserialization() -> None:
    manifest = Manifest(
        entries={
            "credentials.json": LocalFile(src=Path("trusted/credentials.json")),
            "data": GCSMount(
                bucket="bucket",
                service_account_file="/workspace/credentials.json",
                mount_strategy=DockerVolumeMountStrategy(driver="rclone"),
            ),
        }
    )
    payload: dict[str, object] = {
        "type": "test",
        "manifest": manifest.model_dump(mode="json"),
        "snapshot": NoopSnapshot(id="snapshot").model_dump(mode="json"),
    }

    with pytest.raises(ValueError, match="sandbox session state payload is invalid"):
        _SecurityTestClient().deserialize_session_state(payload)

    assert payload == {}


def test_raw_state_sanitization_rejects_unknown_pattern_discriminator() -> None:
    sentinel = "unknown-pattern-secret"
    manifest = Manifest(
        entries={
            "data": _s3_mount(strategy=InContainerMountStrategy(pattern=RcloneMountPattern())),
        }
    ).model_dump(mode="json")
    manifest["entries"]["data"]["mount_strategy"]["pattern"]["type"] = sentinel
    payload: dict[str, object] = {"manifest": manifest}

    with pytest.raises(ValueError, match="unknown type") as exc_info:
        sanitize_raw_session_state_mount_authority(payload)

    assert sentinel not in str(exc_info.value)


def test_raw_state_rejects_registered_custom_mount_before_validation() -> None:
    class CustomValidatedS3Mount(S3Mount):
        type: Literal["custom_validated_s3_mount"] = "custom_validated_s3_mount"  # type: ignore[assignment]
        validator_called: ClassVar[bool] = False

        @model_validator(mode="before")
        @classmethod
        def _record_validation(cls, value: Any) -> Any:
            cls.validator_called = True
            return value

    payload: dict[str, object] = {
        "type": "test",
        "manifest": {
            "entries": {
                "data": {
                    "type": "custom_validated_s3_mount",
                    "bucket": "example-bucket",
                    "access_key_id": "access-key",
                    "secret_access_key": "custom-validator-secret",
                    "mount_strategy": {
                        "type": "in_container",
                        "pattern": {"type": "rclone"},
                    },
                }
            }
        },
        "snapshot": NoopSnapshot(id="snapshot").model_dump(mode="json"),
    }

    with pytest.raises(ValueError, match="sandbox session state payload is invalid") as exc_info:
        _SecurityTestClient().deserialize_session_state(payload)

    assert CustomValidatedS3Mount.validator_called is False
    assert payload == {}
    assert "custom-validator-secret" not in str(exc_info.value)


def test_raw_state_rejects_replaced_strategy_registry_before_validation() -> None:
    class CustomValidatedStrategy(InContainerMountStrategy):
        type: Literal["custom_validated_strategy"] = "custom_validated_strategy"  # type: ignore[assignment]
        validator_called: ClassVar[bool] = False

        @model_validator(mode="before")
        @classmethod
        def _record_validation(cls, value: Any) -> Any:
            cls.validator_called = True
            return value

    original_class = MountStrategyBase._subclass_registry["in_container"]
    MountStrategyBase._subclass_registry["in_container"] = CustomValidatedStrategy
    payload: dict[str, object] = {
        "type": "test",
        "manifest": {
            "entries": {
                "data": {
                    "type": "s3_mount",
                    "bucket": "example-bucket",
                    "mount_strategy": {
                        "type": "in_container",
                        "pattern": {"type": "rclone"},
                    },
                }
            }
        },
        "snapshot": NoopSnapshot(id="snapshot").model_dump(mode="json"),
    }
    try:
        with pytest.raises(
            ValueError, match="sandbox session state payload is invalid"
        ) as exc_info:
            _SecurityTestClient().deserialize_session_state(payload)
    finally:
        MountStrategyBase._subclass_registry["in_container"] = original_class
        MountStrategyBase._subclass_registry.pop("custom_validated_strategy", None)

    assert CustomValidatedStrategy.validator_called is False
    assert payload == {}
    assert "custom-validator-secret" not in str(exc_info.value)


def test_raw_state_rejects_malformed_credential_file_locator() -> None:
    manifest = Manifest(
        entries={
            "credentials.json": File(content=b"credential-file-secret"),
            "data": GCSMount(
                bucket="bucket",
                service_account_file="/workspace/credentials.json",
                mount_strategy=DockerVolumeMountStrategy(driver="rclone"),
            ),
        }
    ).model_dump(mode="json")
    manifest["entries"]["data"]["service_account_file"] = ["credentials.json"]
    payload: dict[str, object] = {
        "type": "test",
        "manifest": manifest,
        "snapshot": NoopSnapshot(id="snapshot").model_dump(mode="json"),
    }

    with pytest.raises(ValueError, match="sandbox session state payload is invalid") as exc:
        _SecurityTestClient().deserialize_session_state(payload)

    assert payload == {}
    assert "credential-file-secret" not in str(exc.value)


@pytest.mark.asyncio
async def test_operation_error_with_mount_authority_is_replaced() -> None:
    sentinel = "provider-operation-secret"
    manifest = Manifest(
        entries={
            "data": S3Mount(
                bucket="bucket",
                access_key_id="access-key",
                secret_access_key=sentinel,
                mount_strategy=DockerVolumeMountStrategy(driver="rclone"),
            )
        }
    )

    class HostileProviderError(RuntimeError):
        pass

    _install_hostile_exception_descriptors(HostileProviderError)
    provider_error = HostileProviderError(f"provider failed with {sentinel}")

    @redact_mount_error_data
    async def fail(*, manifest: Manifest) -> None:
        _ = manifest
        raise provider_error

    with pytest.raises(RuntimeError, match="protected mount configuration") as exc:
        await fail(manifest=manifest)

    assert type(exc.value) is RuntimeError
    assert sentinel not in str(exc.value)
    assert exc.value.__cause__ is None
    assert exc.value.__context__ is None
    assert cast(Any, BaseException.args).__get__(provider_error, type(provider_error)) == ()
    assert (
        cast(Any, BaseException.__traceback__).__get__(provider_error, type(provider_error)) is None
    )
    traceback = exc.value.__traceback__
    while traceback is not None:
        frame_path = Path(traceback.tb_frame.f_code.co_filename).as_posix()
        if "/src/agents/" in frame_path:
            assert sentinel not in repr(traceback.tb_frame.f_locals)
        traceback = traceback.tb_next


@pytest.mark.parametrize(
    "mount",
    [
        AzureBlobMount(
            account="example",
            container="private",
            mount_strategy=InContainerMountStrategy(pattern=FuseMountPattern()),
        ),
        S3FilesMount(
            file_system_id="fs-123",
            mount_strategy=InContainerMountStrategy(pattern=S3FilesMountPattern()),
        ),
    ],
)
@pytest.mark.asyncio
async def test_operation_error_with_implicit_broad_authority_is_replaced(
    mount: Mount,
) -> None:
    sentinel = "implicit-broad-provider-secret"
    manifest = Manifest(
        entries={"data": mount}
    ).with_in_container_mount_broad_credential_exposure_acknowledged("data")
    provider_error = RuntimeError(sentinel)

    @redact_mount_error_data
    async def fail(*, manifest: Manifest) -> None:
        _ = manifest
        raise provider_error

    with pytest.raises(RuntimeError, match="protected mount configuration") as exc:
        await fail(manifest=manifest)

    assert sentinel not in str(exc.value)
    assert exc.value.__cause__ is None
    assert exc.value.__context__ is None
    assert provider_error.args == ()
    assert provider_error.__traceback__ is None
    traceback = exc.value.__traceback__
    while traceback is not None:
        frame_path = Path(traceback.tb_frame.f_code.co_filename).as_posix()
        if "/src/agents/" in frame_path:
            assert sentinel not in repr(traceback.tb_frame.f_locals)
        traceback = traceback.tb_next


@pytest.mark.asyncio
async def test_forged_redaction_marker_cannot_return_provider_error() -> None:
    sentinel = "forged-provider-marker-secret"
    manifest = Manifest(
        entries={
            "data": S3Mount(
                bucket="bucket",
                access_key_id="access-key",
                secret_access_key=sentinel,
                mount_strategy=DockerVolumeMountStrategy(driver="rclone"),
            )
        }
    )
    provider_error = RuntimeError(sentinel)
    provider_error._agents_data_redacted = True  # type: ignore[attr-defined]

    @redact_mount_error_data
    async def fail(*, manifest: Manifest) -> None:
        _ = manifest
        raise provider_error

    with pytest.raises(RuntimeError, match="protected mount configuration") as exc_info:
        await fail(manifest=manifest)

    assert exc_info.value is not provider_error
    assert provider_error.args == ()
    assert provider_error.__dict__ == {}
    assert provider_error.__traceback__ is None
    assert sentinel not in repr(exc_info.value)


@pytest.mark.parametrize("error_kind", ["config", "command"])
@pytest.mark.asyncio
async def test_sdk_raise_helper_sanitizes_external_structured_mount_error(
    error_kind: str,
) -> None:
    sentinel = f"forged-sdk-helper-{error_kind}-secret"
    manifest = Manifest(
        entries={
            "data": S3Mount(
                bucket="bucket",
                access_key_id="access-key",
                secret_access_key=sentinel,
                mount_strategy=DockerVolumeMountStrategy(driver="rclone"),
            )
        }
    )
    source_error: MountConfigError | MountCommandError
    if error_kind == "config":
        source_error = MountConfigError(message=sentinel)
    else:
        source_error = MountCommandError(
            command=sentinel,
            stderr=sentinel,
            context={"provider": sentinel},
        )

    @redact_mount_error_data
    async def fail(*, manifest: Manifest) -> None:
        _ = manifest
        _mark_error_data_redacted(source_error)
        _raise_data_redacted_error(source_error)

    expected_error = MountConfigError if error_kind == "config" else MountCommandError
    with pytest.raises(expected_error) as exc_info:
        await fail(manifest=manifest)

    if error_kind == "config":
        assert type(exc_info.value) is MountConfigError
        assert str(exc_info.value) == "sandbox mount configuration is invalid"
    else:
        _assert_sanitized_mount_command_error(exc_info.value)
    assert source_error.args == ()
    assert source_error.__dict__ == {}
    assert source_error.__traceback__ is None
    assert sentinel not in repr(exc_info.value)


@pytest.mark.parametrize("boundary", ["async", "sync"])
@pytest.mark.asyncio
async def test_mount_tool_missing_error_preserves_safe_subtype_across_redaction(
    boundary: str,
) -> None:
    sentinel = f"mount-tool-missing-{boundary}-secret"
    manifest = Manifest(
        entries={
            "data": S3Mount(
                bucket="bucket",
                access_key_id="access-key",
                secret_access_key=sentinel,
                mount_strategy=DockerVolumeMountStrategy(driver="rclone"),
            )
        }
    )
    source_error = MountToolMissingError(
        tool=sentinel,
        context={"provider": sentinel},
        cause=RuntimeError(sentinel),
    )

    @redact_mount_error_data
    async def fail(*, manifest: Manifest) -> None:
        _ = manifest
        raise source_error

    @redact_mount_error_data_sync
    def fail_sync(*, manifest: Manifest) -> None:
        _ = manifest
        raise source_error

    with pytest.raises(MountToolMissingError) as exc_info:
        if boundary == "async":
            await fail(manifest=manifest)
        else:
            fail_sync(manifest=manifest)

    _assert_sanitized_mount_tool_missing_error(exc_info.value)
    assert source_error.args == ()
    assert source_error.__dict__ == {}
    assert source_error.__traceback__ is None
    assert sentinel not in repr(exc_info.value)


@pytest.mark.parametrize("error_kind", ["config", "command"])
@pytest.mark.asyncio
async def test_foreign_code_with_sdk_globals_and_filename_cannot_authorize_mount_error(
    error_kind: str,
) -> None:
    sentinel = f"foreign-sdk-code-{error_kind}-secret"
    manifest = Manifest(
        entries={
            "data": S3Mount(
                bucket="bucket",
                access_key_id="access-key",
                secret_access_key=sentinel,
                mount_strategy=DockerVolumeMountStrategy(driver="rclone"),
            )
        }
    )
    module = importlib.import_module("agents.sandbox._mount_security")
    statement = (
        f"raise MountConfigError(message={sentinel!r})"
        if error_kind == "config"
        else "raise MountCommandError("
        f"command={sentinel!r}, stderr={sentinel!r}, context={{'provider': {sentinel!r}}})"
    )
    foreign_globals = {
        **module.__dict__,
        "MountCommandError": MountCommandError,
        "MountConfigError": MountConfigError,
    }
    try:
        exec(compile(statement, cast(str, module.__file__), "exec"), foreign_globals)
    except (MountConfigError, MountCommandError) as error:
        source_error = error
    _mark_error_data_redacted(source_error)

    @redact_mount_error_data
    async def fail(*, manifest: Manifest) -> None:
        _ = manifest
        raise source_error

    expected_error = MountConfigError if error_kind == "config" else MountCommandError
    with pytest.raises(expected_error) as exc_info:
        await fail(manifest=manifest)

    if error_kind == "config":
        assert type(exc_info.value) is MountConfigError
        assert str(exc_info.value) == "sandbox mount configuration is invalid"
    else:
        _assert_sanitized_mount_command_error(exc_info.value)
    assert source_error.args == ()
    assert source_error.__dict__ == {}
    assert source_error.__traceback__ is None
    assert sentinel not in repr(exc_info.value)


def test_reassigned_core_producer_cannot_authorize_mount_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = "reassigned-core-producer-secret"
    manifest = Manifest(
        entries={
            "data": S3Mount(
                bucket="bucket",
                access_key_id="access-key",
                secret_access_key=sentinel,
                mount_strategy=DockerVolumeMountStrategy(driver="rclone"),
            )
        }
    )
    state = TestSessionState(
        manifest=manifest,
        snapshot=NoopSnapshot(id="snapshot"),
    )
    source_error = MountCommandError(
        command=sentinel,
        stderr=sentinel,
        context={"provider": sentinel},
    )

    class ForeignProducer:
        def __call__(self, client: object, session_state: object) -> object:
            _ = (client, session_state)
            _mark_error_data_redacted(source_error)
            raise source_error

    monkeypatch.setattr(BaseSandboxClient, "serialize_session_state", ForeignProducer())

    @redact_mount_error_data_sync
    def serialize(*, manifest: Manifest) -> object:
        _ = manifest
        return BaseSandboxClient.serialize_session_state(_SecurityTestClient(), state)

    with pytest.raises(MountCommandError) as exc_info:
        serialize(manifest=manifest)

    _assert_sanitized_mount_command_error(exc_info.value)
    assert source_error.args == ()
    assert source_error.__dict__ == {}
    assert source_error.__traceback__ is None
    assert sentinel not in repr(exc_info.value)


@pytest.mark.parametrize("boundary", ["async", "sync"])
@pytest.mark.parametrize("error_kind", ["config", "command"])
@pytest.mark.asyncio
async def test_malformed_structured_mount_error_falls_back_to_generic_redaction(
    boundary: str,
    error_kind: str,
) -> None:
    sentinel = f"malformed-{boundary}-{error_kind}-classification-secret"
    manifest = Manifest(
        entries={
            "data": S3Mount(
                bucket="bucket",
                access_key_id="access-key",
                secret_access_key=sentinel,
                mount_strategy=DockerVolumeMountStrategy(driver="rclone"),
            )
        }
    )
    source_error: MountConfigError | MountCommandError
    if error_kind == "config":
        source_error = MountConfigError(message=sentinel)
    else:
        source_error = MountCommandError(
            command=sentinel,
            stderr=sentinel,
            retryable=False,
        )
    source_error.retryable = cast(Any, sentinel)

    @redact_mount_error_data
    async def fail(*, manifest: Manifest) -> None:
        _ = manifest
        raise source_error

    @redact_mount_error_data_sync
    def fail_sync(*, manifest: Manifest) -> None:
        _ = manifest
        raise source_error

    with pytest.raises(RuntimeError, match="protected mount configuration") as exc_info:
        if boundary == "async":
            await fail(manifest=manifest)
        else:
            fail_sync(manifest=manifest)

    assert type(exc_info.value) is RuntimeError
    assert source_error.args == ()
    assert source_error.__dict__ == {}
    assert source_error.__traceback__ is None
    assert sentinel not in repr(exc_info.value)


@pytest.mark.parametrize("boundary", ["async", "sync"])
@pytest.mark.parametrize("hostile_descriptor", ["__class__", "__dict__"])
@pytest.mark.asyncio
async def test_protected_mount_redaction_ignores_hostile_identity_and_state_descriptors(
    boundary: str,
    hostile_descriptor: str,
) -> None:
    sentinel = f"hostile-{boundary}-{hostile_descriptor}-secret"
    manifest = Manifest(
        entries={
            "data": S3Mount(
                bucket="bucket",
                access_key_id="access-key",
                secret_access_key=sentinel,
                mount_strategy=DockerVolumeMountStrategy(driver="rclone"),
            )
        }
    )
    descriptor_calls: list[str] = []

    def reject_descriptor_access(_error: BaseException) -> object:
        descriptor_calls.append(hostile_descriptor)
        raise AssertionError(f"provider-defined {hostile_descriptor} descriptor was accessed")

    expected_error: type[BaseException]
    if hostile_descriptor == "__class__":
        hostile_error_type = type(
            "HostileProviderError",
            (RuntimeError,),
            {"__class__": property(reject_descriptor_access)},
        )
        source_error: BaseException = hostile_error_type(sentinel)
        expected_error = RuntimeError
    else:
        hostile_error_type = type(
            "HostileStructuredError",
            (SandboxError,),
            {"__dict__": property(reject_descriptor_access)},
        )
        source_error = hostile_error_type(
            message=sentinel,
            error_code=ErrorCode.MOUNT_FAILED,
            op="materialize",
            context={"provider": sentinel},
            retryable=True,
        )
        expected_error = SandboxError

    @redact_mount_error_data
    async def fail(*, manifest: Manifest) -> None:
        _ = manifest
        raise source_error

    @redact_mount_error_data_sync
    def fail_sync(*, manifest: Manifest) -> None:
        _ = manifest
        raise source_error

    with pytest.raises(expected_error) as exc_info:
        if boundary == "async":
            await fail(manifest=manifest)
        else:
            fail_sync(manifest=manifest)

    assert descriptor_calls == []
    if hostile_descriptor == "__class__":
        assert type(exc_info.value) is RuntimeError
        assert str(exc_info.value) == (
            "sandbox operation failed while using a protected mount configuration"
        )
    else:
        assert type(exc_info.value) is SandboxError
        assert exc_info.value.error_code is ErrorCode.MOUNT_FAILED
        assert exc_info.value.op == "materialize"
        assert exc_info.value.retryable is True
        assert exc_info.value.context == {}
    assert cast(Any, BaseException.args).__get__(source_error, type(source_error)) == ()
    assert cast(Any, BaseException.__traceback__).__get__(source_error, type(source_error)) is None
    assert sentinel not in repr(exc_info.value)


def test_mutated_core_producer_wrapped_link_does_not_change_trust(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = "mutated-core-producer-wrapped-link-secret"

    class CustomMount(S3Mount):
        type: Literal["mutated_core_mount"] = "mutated_core_mount"  # type: ignore[assignment]
        api_token: str = sentinel

    state = TestSessionState(
        manifest=Manifest(
            entries={
                "data": CustomMount(
                    bucket="bucket",
                    mount_strategy=DockerVolumeMountStrategy(driver="rclone"),
                )
            }
        ),
        snapshot=NoopSnapshot(id="snapshot"),
    )

    class ForeignProducer:
        def __call__(self, *args: object, **kwargs: object) -> object:
            _ = (args, kwargs)
            raise AssertionError("the wrapped link must not be invoked")

    monkeypatch.setattr(
        BaseSandboxClient.serialize_session_state,
        "__wrapped__",
        ForeignProducer(),
    )

    with pytest.raises(MountConfigError) as exc_info:
        _SecurityTestClient().serialize_session_state(state)

    assert sentinel not in repr(exc_info.value)


def test_sync_operation_error_with_mount_authority_clears_source_arguments() -> None:
    sentinel = "sync-provider-operation-secret"
    manifest = Manifest(
        entries={
            "data": S3Mount(
                bucket="bucket",
                access_key_id="access-key",
                secret_access_key=sentinel,
                mount_strategy=DockerVolumeMountStrategy(driver="rclone"),
            )
        }
    )

    class HostileProviderError(RuntimeError):
        pass

    _install_hostile_exception_descriptors(HostileProviderError)
    provider_error = HostileProviderError(f"provider failed with {sentinel}")

    @redact_mount_error_data_sync
    def fail(*, manifest: Manifest) -> None:
        _ = manifest
        raise provider_error

    with pytest.raises(RuntimeError, match="protected mount configuration") as exc:
        fail(manifest=manifest)

    assert type(exc.value) is RuntimeError
    assert sentinel not in str(exc.value)
    assert exc.value.__cause__ is None
    assert exc.value.__context__ is None
    assert cast(Any, BaseException.args).__get__(provider_error, type(provider_error)) == ()
    assert (
        cast(Any, BaseException.__traceback__).__get__(provider_error, type(provider_error)) is None
    )


@pytest.mark.parametrize(
    "mount",
    [
        AzureBlobMount(
            account="example",
            container="private",
            mount_strategy=InContainerMountStrategy(pattern=FuseMountPattern()),
        ),
        S3FilesMount(
            file_system_id="fs-123",
            mount_strategy=InContainerMountStrategy(pattern=S3FilesMountPattern()),
        ),
    ],
)
def test_sync_operation_error_with_implicit_broad_authority_is_replaced(
    mount: Mount,
) -> None:
    sentinel = "sync-implicit-broad-provider-secret"
    manifest = Manifest(
        entries={"data": mount}
    ).with_in_container_mount_broad_credential_exposure_acknowledged("data")
    provider_error = RuntimeError(sentinel)

    @redact_mount_error_data_sync
    def fail(*, manifest: Manifest) -> None:
        _ = manifest
        raise provider_error

    with pytest.raises(RuntimeError, match="protected mount configuration") as exc:
        fail(manifest=manifest)

    assert sentinel not in str(exc.value)
    assert exc.value.__cause__ is None
    assert exc.value.__context__ is None
    assert provider_error.args == ()
    assert provider_error.__traceback__ is None


@pytest.mark.asyncio
async def test_protected_mixed_exception_group_is_replaced_and_discarded() -> None:
    sentinel = "mixed-exception-group-secret"
    manifest = Manifest(
        entries={
            "data": S3Mount(
                bucket="bucket",
                access_key_id="access-key",
                secret_access_key=sentinel,
                mount_strategy=DockerVolumeMountStrategy(driver="rclone"),
            )
        }
    )

    child_error: RuntimeError
    try:
        sensitive_local = {"credential": sentinel}
        raise RuntimeError(sensitive_local)
    except RuntimeError as error:
        child_error = error
        child_error.payload = sensitive_local  # type: ignore[attr-defined]
    cancellation = asyncio.CancelledError(sentinel)
    source_group = BaseExceptionGroup(sentinel, [child_error, cancellation])

    @redact_mount_error_data
    async def fail(*, manifest: Manifest) -> None:
        _ = manifest
        raise source_group

    with pytest.raises(RuntimeError, match="protected mount configuration") as exc_info:
        await fail(manifest=manifest)

    assert type(exc_info.value) is RuntimeError
    assert source_group.args[0] == "Error details are redacted."
    assert source_group.__traceback__ is None
    assert child_error.args == ()
    assert child_error.__dict__ == {}
    assert child_error.__traceback__ is None
    assert cancellation.args == ()
    assert sentinel not in repr(exc_info.value)
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    traceback_cursor = exc_info.value.__traceback__
    while traceback_cursor is not None:
        frame_path = Path(traceback_cursor.tb_frame.f_code.co_filename).as_posix()
        if "/src/agents/" in frame_path:
            assert all(
                value is not source_group for value in traceback_cursor.tb_frame.f_locals.values()
            )
            assert sentinel not in repr(traceback_cursor.tb_frame.f_locals)
        traceback_cursor = traceback_cursor.tb_next


@pytest.mark.asyncio
async def test_protected_mount_failure_discards_retained_nested_exception() -> None:
    sentinel = "mount-retained-nested-exception-secret"
    manifest = Manifest(
        entries={
            "data": S3Mount(
                bucket="bucket",
                access_key_id="access-key",
                secret_access_key=sentinel,
                mount_strategy=DockerVolumeMountStrategy(driver="rclone"),
            )
        }
    )
    child_error = RuntimeError(sentinel)
    child_error.payload = {"credential": sentinel}  # type: ignore[attr-defined]
    source_error = RuntimeError("safe provider failure", {"children": [child_error]})
    source_error.payload = (child_error,)  # type: ignore[attr-defined]

    @redact_mount_error_data
    async def fail(*, manifest: Manifest) -> None:
        _ = manifest
        raise source_error

    with pytest.raises(RuntimeError, match="protected mount configuration") as exc_info:
        await fail(manifest=manifest)

    assert source_error.args == ()
    assert source_error.__dict__ == {}
    assert source_error.__traceback__ is None
    assert child_error.args == ()
    assert child_error.__dict__ == {}
    assert child_error.__traceback__ is None
    assert sentinel not in repr(exc_info.value)


def test_sync_protected_exception_group_is_replaced_and_discarded() -> None:
    sentinel = "sync-exception-group-secret"
    manifest = Manifest(
        entries={
            "data": S3Mount(
                bucket="bucket",
                access_key_id="access-key",
                secret_access_key=sentinel,
                mount_strategy=DockerVolumeMountStrategy(driver="rclone"),
            )
        }
    )
    child_error = RuntimeError(sentinel)
    source_group = BaseExceptionGroup(sentinel, [child_error])

    @redact_mount_error_data_sync
    def fail(*, manifest: Manifest) -> None:
        _ = manifest
        raise source_group

    with pytest.raises(RuntimeError, match="protected mount configuration") as exc_info:
        fail(manifest=manifest)

    assert type(exc_info.value) is RuntimeError
    assert source_group.args[0] == "Error details are redacted."
    assert source_group.__traceback__ is None
    assert child_error.args == ()
    assert child_error.__traceback__ is None
    assert sentinel not in repr(exc_info.value)
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None


@pytest.mark.asyncio
async def test_credentialless_exception_group_preserves_raw_diagnostics() -> None:
    source_group = BaseExceptionGroup("credentialless provider failures", [RuntimeError("detail")])

    @redact_mount_error_data
    async def fail() -> None:
        raise source_group

    with pytest.raises(BaseExceptionGroup) as exc_info:
        await fail()

    assert exc_info.value is source_group
    assert source_group.args
    assert source_group.__traceback__ is not None


@pytest.mark.asyncio
async def test_initial_authority_classification_cancellation_is_value_free() -> None:
    sentinel = "initial-classification-cancellation-secret"
    source_cancel = asyncio.CancelledError(sentinel)
    operation_called = False

    class HostileState:
        @property
        def state(self) -> object:
            raise source_cancel

    @redact_mount_error_data
    async def operation(value: object) -> None:
        nonlocal operation_called
        operation_called = True
        _ = value

    with pytest.raises(asyncio.CancelledError) as exc_info:
        await operation(HostileState())

    assert operation_called is False
    assert type(exc_info.value) is asyncio.CancelledError
    assert source_cancel.args == ()
    assert source_cancel.__traceback__ is None
    traceback_cursor = exc_info.value.__traceback__
    while traceback_cursor is not None:
        frame_path = Path(traceback_cursor.tb_frame.f_code.co_filename).as_posix()
        if "/src/agents/" in frame_path:
            assert sentinel not in repr(traceback_cursor.tb_frame.f_locals)
        traceback_cursor = traceback_cursor.tb_next


@pytest.mark.parametrize(
    ("boundary", "failure_stage", "expected_type", "expected_args"),
    [
        ("async", "classification", SystemExit, (1,)),
        ("sync", "operation", KeyboardInterrupt, ()),
    ],
)
@pytest.mark.asyncio
async def test_protected_mount_process_control_is_value_free(
    boundary: str,
    failure_stage: str,
    expected_type: type[BaseException],
    expected_args: tuple[object, ...],
) -> None:
    sentinel = f"{boundary}-{failure_stage}-process-control-secret"

    class ProviderSystemExit(SystemExit):
        pass

    class ProviderKeyboardInterrupt(KeyboardInterrupt):
        pass

    source_error: BaseException = (
        ProviderSystemExit(sentinel)
        if expected_type is SystemExit
        else ProviderKeyboardInterrupt(sentinel)
    )
    operation_called = False

    class HostileAuthorityProbe:
        @property
        def state(self) -> object:
            raise source_error

    manifest = Manifest(
        entries={
            "data": S3Mount(
                bucket="bucket",
                access_key_id="access-key",
                secret_access_key=sentinel,
                mount_strategy=DockerVolumeMountStrategy(driver="rclone"),
            )
        }
    )

    with pytest.raises(expected_type) as exc_info:
        if boundary == "async":

            @redact_mount_error_data
            async def async_operation(value: object) -> None:
                nonlocal operation_called
                operation_called = True
                _ = value

            await async_operation(HostileAuthorityProbe())
        else:

            @redact_mount_error_data_sync
            def sync_operation(*, manifest: Manifest) -> None:
                nonlocal operation_called
                operation_called = True
                _ = manifest
                raise source_error

            sync_operation(manifest=manifest)

    assert operation_called is (failure_stage == "operation")
    assert type(exc_info.value) is expected_type
    assert exc_info.value.args == expected_args
    assert exc_info.value is not source_error
    assert source_error.args == ()
    assert source_error.__traceback__ is None
    assert sentinel not in repr(exc_info.value)


@pytest.mark.parametrize(
    ("argument_code", "effective_code"),
    [(False, True), (True, False)],
)
def test_sync_protected_mount_rejects_inconsistent_system_exit_codes(
    argument_code: bool,
    effective_code: bool,
) -> None:
    source_error = SystemExit(argument_code)
    source_error.code = effective_code
    manifest = Manifest(
        entries={
            "data": S3Mount(
                bucket="bucket",
                access_key_id="access-key",
                secret_access_key="secret-access-key",
                mount_strategy=DockerVolumeMountStrategy(driver="rclone"),
            )
        }
    )

    @redact_mount_error_data_sync
    def fail(*, manifest: Manifest) -> None:
        _ = manifest
        raise source_error

    with pytest.raises(SystemExit) as exc_info:
        fail(manifest=manifest)

    assert type(exc_info.value) is SystemExit
    assert exc_info.value.args == (1,)
    assert exc_info.value is not source_error
    assert source_error.args == ()
    assert source_error.code is None
    assert source_error.__traceback__ is None


def test_sync_initial_authority_classification_group_is_value_free() -> None:
    sentinel = "sync-classification-group-secret"
    child = RuntimeError(sentinel)
    cancellation = asyncio.CancelledError(sentinel)
    source_group = BaseExceptionGroup("classification failed", [child, cancellation])
    operation_called = False

    class HostileAuthorityProbe:
        def _runtime_has_protected_mount_authority(self) -> bool:
            raise source_group

    @redact_mount_error_data_sync
    def operation(value: object) -> None:
        nonlocal operation_called
        operation_called = True
        _ = value

    with pytest.raises(RuntimeError, match="protected mount configuration") as exc_info:
        operation(HostileAuthorityProbe())

    assert operation_called is False
    assert source_group.args[0] == "Error details are redacted."
    assert source_group.__traceback__ is None
    assert child.args == ()
    assert cancellation.args == ()
    assert sentinel not in repr(exc_info.value)


@pytest.mark.parametrize("boundary", ["async", "sync"])
@pytest.mark.asyncio
async def test_initial_authority_classification_exception_stops_before_side_effects(
    boundary: str,
) -> None:
    sentinel = f"{boundary}-classification-exception-secret"
    source_error = RuntimeError(sentinel)
    operation_called = False

    class HostileAuthorityProbe:
        @property
        def state(self) -> object:
            raise source_error

    with pytest.raises(RuntimeError, match="protected mount configuration") as exc_info:
        if boundary == "async":

            @redact_mount_error_data
            async def async_operation(value: object) -> None:
                nonlocal operation_called
                operation_called = True
                _ = value

            await async_operation(HostileAuthorityProbe())
        else:

            @redact_mount_error_data_sync
            def sync_operation(value: object) -> None:
                nonlocal operation_called
                operation_called = True
                _ = value

            sync_operation(HostileAuthorityProbe())

    assert operation_called is False
    assert source_error.args == ()
    assert source_error.__traceback__ is None
    assert sentinel not in repr(exc_info.value)


@pytest.mark.parametrize("operation", ["activate", "deactivate"])
@pytest.mark.asyncio
async def test_direct_docker_volume_lifecycle_call_clears_mount_frames(
    operation: str,
) -> None:
    sentinel = f"direct-docker-{operation}-secret"
    mount = S3Mount(
        bucket="bucket",
        access_key_id="access-key",
        secret_access_key=sentinel,
        mount_strategy=DockerVolumeMountStrategy(driver="rclone"),
    )
    strategy = cast(DockerVolumeMountStrategy, mount.mount_strategy)

    class UnsupportedSession:
        def supports_docker_volume_mounts(self) -> bool:
            return False

    session = cast(Any, UnsupportedSession())
    lifecycle = getattr(strategy, operation)
    with pytest.raises(MountConfigError) as exc_info:
        await lifecycle(mount, session, Path("/workspace/data"), Path("/workspace"))

    assert sentinel not in repr(exc_info.value)
    traceback_cursor = exc_info.value.__traceback__
    while traceback_cursor is not None:
        frame_path = Path(traceback_cursor.tb_frame.f_code.co_filename).as_posix()
        if "/src/agents/" in frame_path:
            assert sentinel not in repr(traceback_cursor.tb_frame.f_locals)
        traceback_cursor = traceback_cursor.tb_next


@pytest.mark.asyncio
async def test_failure_edge_classification_cancellation_discards_both_errors() -> None:
    sentinel = "failure-edge-classification-secret"
    source_cancel = asyncio.CancelledError(sentinel)
    operation_error = RuntimeError(sentinel)

    class MutableAuthorityProbe:
        fail_classification = False

        @property
        def state(self) -> object | None:
            if self.fail_classification:
                raise source_cancel
            return None

    @redact_mount_error_data
    async def operation(value: MutableAuthorityProbe) -> None:
        value.fail_classification = True
        raise operation_error

    with pytest.raises(asyncio.CancelledError) as exc_info:
        await operation(MutableAuthorityProbe())

    assert type(exc_info.value) is asyncio.CancelledError
    assert source_cancel.args == ()
    assert source_cancel.__traceback__ is None
    assert operation_error.args == ()
    assert operation_error.__traceback__ is None
    traceback_cursor = exc_info.value.__traceback__
    while traceback_cursor is not None:
        frame_path = Path(traceback_cursor.tb_frame.f_code.co_filename).as_posix()
        if "/src/agents/" in frame_path:
            assert sentinel not in repr(traceback_cursor.tb_frame.f_locals)
        traceback_cursor = traceback_cursor.tb_next


@pytest.mark.parametrize(
    "boundary",
    ["async-operation", "sync-operation", "async-classification", "sync-classification"],
)
@pytest.mark.asyncio
async def test_direct_base_exception_is_redacted_at_protected_mount_boundaries(
    boundary: str,
) -> None:
    class ProviderAbort(BaseException):
        pass

    sentinel = f"{boundary}-base-exception-secret"
    source_error = ProviderAbort(sentinel)
    manifest = Manifest(
        entries={
            "data": S3Mount(
                bucket="bucket",
                access_key_id="access-key",
                secret_access_key=sentinel,
                mount_strategy=DockerVolumeMountStrategy(driver="rclone"),
            )
        }
    )

    class HostileAuthorityProbe:
        @property
        def state(self) -> object:
            raise source_error

    with pytest.raises(RuntimeError, match="protected mount configuration") as exc_info:
        if boundary == "async-operation":

            @redact_mount_error_data
            async def fail_async_operation(*, manifest: Manifest) -> None:
                _ = manifest
                raise source_error

            await fail_async_operation(manifest=manifest)
        elif boundary == "sync-operation":

            @redact_mount_error_data_sync
            def fail_sync_operation(*, manifest: Manifest) -> None:
                _ = manifest
                raise source_error

            fail_sync_operation(manifest=manifest)
        elif boundary == "async-classification":

            @redact_mount_error_data
            async def fail_async_classification(value: object) -> None:
                _ = value
                raise AssertionError("operation must not run")

            await fail_async_classification(HostileAuthorityProbe())
        else:

            @redact_mount_error_data_sync
            def fail_sync_classification(value: object) -> None:
                _ = value
                raise AssertionError("operation must not run")

            fail_sync_classification(HostileAuthorityProbe())

    assert type(exc_info.value) is RuntimeError
    assert source_error.args == ()
    assert source_error.__traceback__ is None
    assert sentinel not in repr(exc_info.value)
    traceback_cursor = exc_info.value.__traceback__
    while traceback_cursor is not None:
        frame_path = Path(traceback_cursor.tb_frame.f_code.co_filename).as_posix()
        if "/src/agents/" in frame_path:
            assert sentinel not in repr(traceback_cursor.tb_frame.f_locals)
        traceback_cursor = traceback_cursor.tb_next


@pytest.mark.parametrize("operation", ["unmount", "snapshot-teardown"])
@pytest.mark.asyncio
async def test_protected_cleanup_failures_clear_all_mount_lifecycle_frames(
    operation: str,
) -> None:
    sentinel = f"{operation}-lifecycle-secret"
    mount = S3Mount(
        bucket="bucket",
        access_key_id="access-key",
        secret_access_key=sentinel,
        mount_strategy=InContainerMountStrategy(pattern=RcloneMountPattern()),
    )
    state = TestSessionState(
        manifest=Manifest(entries={"data": mount}),
        snapshot=NoopSnapshot(id="snapshot"),
    )

    class FailingCleanupSession:
        def __init__(self) -> None:
            self.state = state
            self.error: RuntimeError | None = None

        async def exec(self, *command: object, **kwargs: object) -> None:
            _ = (command, kwargs)
            provider_local = {"credential": sentinel}
            self.error = RuntimeError(provider_local)
            raise self.error

    session = FailingCleanupSession()

    with pytest.raises(RuntimeError, match="protected mount configuration") as exc_info:
        if operation == "unmount":
            await mount.unmount(cast(Any, session), Path("data"), Path("/"))
        else:
            await mount.mount_strategy.teardown_for_snapshot(
                mount,
                cast(Any, session),
                Path("/workspace/data"),
            )

    assert session.error is not None
    assert session.error.args == ()
    assert session.error.__traceback__ is None
    assert sentinel not in repr(exc_info.value)
    traceback_cursor = exc_info.value.__traceback__
    while traceback_cursor is not None:
        frame_path = Path(traceback_cursor.tb_frame.f_code.co_filename).as_posix()
        if "/src/agents/" in frame_path:
            assert sentinel not in repr(traceback_cursor.tb_frame.f_locals)
        traceback_cursor = traceback_cursor.tb_next


@pytest.mark.asyncio
async def test_unix_local_delete_logs_only_redacted_cleanup_failure(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    unix_local = pytest.importorskip(
        "agents.sandbox.sandboxes.unix_local",
        exc_type=ImportError,
    )
    sentinel = "unix-delete-cleanup-secret"
    mount = S3Mount(
        bucket="bucket",
        access_key_id="access-key",
        secret_access_key=sentinel,
        mount_strategy=InContainerMountStrategy(pattern=RcloneMountPattern()),
    )
    state = unix_local.UnixLocalSandboxSessionState(
        manifest=Manifest(root=str(tmp_path), entries={"data": mount}),
        snapshot=NoopSnapshot(id="snapshot"),
        workspace_root_owned=True,
    )
    inner = unix_local.UnixLocalSandboxSession.from_state(state)
    provider_errors: list[RuntimeError] = []

    async def fail_exec(*command: object, **kwargs: object) -> None:
        _ = (command, kwargs)
        provider_local = {"credential": sentinel}
        error = RuntimeError(provider_local)
        provider_errors.append(error)
        raise error

    monkeypatch.setattr(inner, "_exec_internal", fail_exec)
    client = unix_local.UnixLocalSandboxClient()
    session = client._wrap_session(inner)
    caplog.set_level(logging.WARNING, logger=unix_local.__name__)

    returned = await client.delete(session)

    assert returned is session
    assert tmp_path.exists()
    assert provider_errors
    assert all(error.args == () and error.__traceback__ is None for error in provider_errors)
    assert caplog.records
    for record in caplog.records:
        assert sentinel not in record.getMessage()
        assert sentinel not in repr(record.args)
        assert sentinel not in repr(record.__dict__)


def test_hostile_redaction_marker_lookup_cannot_escape_sync_boundary() -> None:
    sentinel = "hostile-provider-marker-secret"
    manifest = Manifest(
        entries={
            "data": S3Mount(
                bucket="bucket",
                access_key_id="access-key",
                secret_access_key=sentinel,
                mount_strategy=DockerVolumeMountStrategy(driver="rclone"),
            )
        }
    )

    class HostileMarkerError(RuntimeError):
        @property
        def _agents_data_redacted(self) -> bool:
            raise AssertionError("provider marker descriptor was accessed")

    provider_error = HostileMarkerError(sentinel)

    @redact_mount_error_data_sync
    def fail(*, manifest: Manifest) -> None:
        _ = manifest
        raise provider_error

    with pytest.raises(RuntimeError, match="protected mount configuration") as exc_info:
        fail(manifest=manifest)

    assert type(exc_info.value) is RuntimeError
    assert provider_error.args == ()
    assert provider_error.__traceback__ is None
    assert sentinel not in repr(exc_info.value)


@pytest.mark.asyncio
async def test_operation_error_with_read_only_provider_attributes_is_replaced() -> None:
    sentinel = "read-only-provider-secret"
    manifest = Manifest(
        entries={
            "data": S3Mount(
                bucket="bucket",
                access_key_id="access-key",
                secret_access_key=sentinel,
                mount_strategy=DockerVolumeMountStrategy(driver="rclone"),
            )
        }
    )

    class ReadOnlyProviderError(Exception):
        @property
        def context(self) -> str:
            return sentinel

        @property
        def cause(self) -> str:
            return sentinel

    provider_error = ReadOnlyProviderError(sentinel)

    @redact_mount_error_data
    async def fail(*, manifest: Manifest) -> None:
        _ = manifest
        raise provider_error

    with pytest.raises(RuntimeError, match="protected mount configuration") as exc:
        await fail(manifest=manifest)

    assert sentinel not in str(exc.value)
    assert exc.value.__cause__ is None
    assert exc.value.__context__ is None
    assert provider_error.__traceback__ is None
    traceback_cursor = exc.value.__traceback__
    while traceback_cursor is not None:
        frame_path = Path(traceback_cursor.tb_frame.f_code.co_filename).as_posix()
        if "/src/agents/" in frame_path:
            assert sentinel not in repr(traceback_cursor.tb_frame.f_locals)
        traceback_cursor = traceback_cursor.tb_next


@pytest.mark.asyncio
async def test_cancellation_with_mount_authority_preserves_redacted_cancellation() -> None:
    sentinel = "cancelled-provider-secret"
    manifest = Manifest(
        entries={
            "data": S3Mount(
                bucket="bucket",
                access_key_id="access-key",
                secret_access_key=sentinel,
                mount_strategy=DockerVolumeMountStrategy(driver="rclone"),
            )
        }
    )

    class HostileCancelledError(asyncio.CancelledError):
        pass

    _install_hostile_exception_descriptors(HostileCancelledError)
    provider_error = HostileCancelledError(sentinel)

    @redact_mount_error_data
    async def cancel(*, manifest: Manifest) -> None:
        _ = manifest
        raise provider_error

    with pytest.raises(asyncio.CancelledError) as exc:
        await cancel(manifest=manifest)

    assert type(exc.value) is asyncio.CancelledError
    assert sentinel not in str(exc.value)
    assert exc.value.__cause__ is None
    assert exc.value.__context__ is None
    assert cast(Any, BaseException.args).__get__(provider_error, type(provider_error)) == ()
    assert (
        cast(Any, BaseException.__traceback__).__get__(provider_error, type(provider_error)) is None
    )
    traceback = exc.value.__traceback__
    while traceback is not None:
        frame_path = Path(traceback.tb_frame.f_code.co_filename).as_posix()
        if "/src/agents/" in frame_path:
            assert sentinel not in repr(traceback.tb_frame.f_locals)
        traceback = traceback.tb_next


@pytest.mark.asyncio
async def test_structured_cancellation_with_mount_authority_preserves_cancellation() -> None:
    sentinel = "structured-cancellation-secret"
    manifest = Manifest(
        entries={
            "data": S3Mount(
                bucket="bucket",
                access_key_id="access-key",
                secret_access_key=sentinel,
                mount_strategy=DockerVolumeMountStrategy(driver="rclone"),
            )
        }
    )

    class StructuredCancelledError(SandboxError, asyncio.CancelledError):
        pass

    provider_error = StructuredCancelledError(
        message=sentinel,
        error_code=ErrorCode.MOUNT_FAILED,
        op="materialize",
        context={"credential": sentinel},
        retryable=True,
    )

    @redact_mount_error_data
    async def cancel(*, manifest: Manifest) -> None:
        _ = manifest
        raise provider_error

    with pytest.raises(asyncio.CancelledError) as exc_info:
        await cancel(manifest=manifest)

    assert type(exc_info.value) is asyncio.CancelledError
    assert sentinel not in str(exc_info.value)
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert cast(Any, BaseException.args).__get__(provider_error, type(provider_error)) == ()
    assert (
        cast(Any, BaseException.__traceback__).__get__(provider_error, type(provider_error)) is None
    )


def test_generic_session_state_parser_sanitizes_legacy_mount_authority() -> None:
    sentinel = "legacy-session-state-secret"
    manifest = Manifest(
        entries={
            "data": S3Mount(
                bucket="bucket",
                access_key_id="access-key",
                secret_access_key=sentinel,
                mount_strategy=DockerVolumeMountStrategy(driver="rclone"),
            )
        }
    )
    payload: dict[str, object] = {
        "type": "test",
        "manifest": manifest.model_dump(mode="json"),
        "snapshot": NoopSnapshot(id="snapshot").model_dump(mode="json"),
    }

    restored = SandboxSessionState.parse(payload)

    assert restored.mount_authority_redacted is True
    mount = restored.manifest.entries["data"]
    assert isinstance(mount, S3Mount)
    assert mount.access_key_id is None
    assert mount.secret_access_key is None
    assert sentinel not in repr(restored)


def test_direct_session_state_round_trip_redacts_mount_authority() -> None:
    sentinel = "direct-state-secret"
    state = TestSessionState(
        manifest=Manifest(
            entries={
                "data": S3Mount(
                    bucket="bucket",
                    access_key_id="access-key",
                    secret_access_key=sentinel,
                    mount_strategy=DockerVolumeMountStrategy(driver="rclone"),
                )
            }
        ),
        snapshot=NoopSnapshot(id="snapshot"),
    )

    payload = state.model_dump_json()
    restored = TestSessionState.model_validate_json(payload)

    assert sentinel not in payload
    assert REDACTED_MOUNT_AUTHORITY_KEY in payload
    assert restored.mount_authority_redacted is True
    with pytest.raises(ValueError, match="requires a current trusted manifest"):
        restored.rebind_persisted_mount_authority(None, provider_backend_id="docker")


def test_raw_state_sanitization_clears_pattern_authority() -> None:
    manifest = Manifest(
        entries={
            "credentials.conf": File(content=b"credential-file-secret"),
            "rclone": _s3_mount(
                strategy=InContainerMountStrategy(
                    pattern=RcloneMountPattern(
                        extra_args=["--config", "/workspace/credentials.conf"]
                    )
                )
            ),
            "s3files": S3FilesMount(
                file_system_id="fs-123",
                mount_strategy=InContainerMountStrategy(
                    pattern=S3FilesMountPattern(
                        options=S3FilesMountPattern.S3FilesOptions(
                            extra_options={
                                "tlsport": "4049",
                                "secret_access_key": "pattern-secret",
                            }
                        )
                    )
                ),
            ),
        }
    )
    payload: dict[str, object] = {
        "type": "test",
        "manifest": manifest.model_dump(mode="json"),
        "snapshot": NoopSnapshot(id="snapshot").model_dump(mode="json"),
    }

    sanitized, redacted = sanitize_raw_session_state_mount_authority(payload)

    assert redacted is True
    assert isinstance(sanitized, dict)
    entries = sanitized["manifest"]["entries"]
    assert entries["credentials.conf"]["content"] == ""
    assert entries["rclone"]["mount_strategy"]["pattern"]["extra_args"] == []
    assert entries["s3files"]["mount_strategy"]["pattern"]["options"]["extra_options"] == {}
    assert "credential-file-secret" not in repr(sanitized)
    assert "pattern-secret" not in repr(sanitized)


@pytest.mark.parametrize("location", ["strategy", "pattern"])
def test_raw_state_sanitization_rejects_unknown_nested_discriminators(
    location: str,
) -> None:
    sentinel = f"unknown-{location}-secret"
    manifest = Manifest(
        entries={
            "docker": S3Mount(
                bucket="bucket",
                mount_strategy=DockerVolumeMountStrategy(
                    driver="rclone",
                    driver_options={"password": "driver-secret"},
                ),
            ),
            "s3files": S3FilesMount(
                file_system_id="fs-123",
                mount_strategy=InContainerMountStrategy(
                    pattern=S3FilesMountPattern(
                        options=S3FilesMountPattern.S3FilesOptions(
                            extra_options={"password": "pattern-secret"}
                        )
                    )
                ),
            ),
        }
    ).model_dump(mode="json")
    if location == "strategy":
        manifest["entries"]["docker"]["mount_strategy"]["type"] = sentinel
    else:
        manifest["entries"]["s3files"]["mount_strategy"]["pattern"]["type"] = sentinel

    with pytest.raises(ValueError, match="unknown type") as exc_info:
        sanitize_raw_session_state_mount_authority({"type": "test", "manifest": manifest})

    assert sentinel not in str(exc_info.value)


def test_raw_state_sanitization_strips_opaque_fields_with_known_strategy_type() -> None:
    manifest = Manifest(
        entries={
            "data": S3Mount(
                bucket="bucket",
                mount_strategy=DockerVolumeMountStrategy(driver="rclone"),
            )
        }
    ).model_dump(mode="json")
    manifest["entries"]["data"]["mount_strategy"]["api_token"] = "raw-strategy-secret"

    sanitized, redacted = sanitize_raw_session_state_mount_authority(
        {"type": "test", "manifest": manifest}
    )

    strategy = sanitized["manifest"]["entries"]["data"]["mount_strategy"]  # type: ignore[index]
    assert redacted is True
    assert "api_token" not in strategy
    assert "raw-strategy-secret" not in repr(sanitized)


def test_raw_state_sanitization_strips_opaque_nested_pattern_fields() -> None:
    manifest = Manifest(
        entries={
            "data": S3Mount(
                bucket="bucket",
                mount_strategy=InContainerMountStrategy(pattern=RcloneMountPattern()),
            )
        }
    ).model_dump(mode="json")
    pattern = manifest["entries"]["data"]["mount_strategy"]["pattern"]
    pattern["api_token"] = "nested-pattern-secret"
    pattern["options"] = {"authorization": "nested-options-secret"}

    sanitized, redacted = sanitize_raw_session_state_mount_authority(
        {"type": "test", "manifest": manifest}
    )

    sanitized_pattern = sanitized["manifest"]["entries"]["data"]["mount_strategy"]["pattern"]  # type: ignore[index]
    assert redacted is True
    assert "api_token" not in sanitized_pattern
    assert "options" not in sanitized_pattern
    assert "nested-pattern-secret" not in repr(sanitized)
    assert "nested-options-secret" not in repr(sanitized)


def test_deserialization_sanitizes_input_before_validation_errors() -> None:
    manifest = Manifest(
        entries={
            "data": _s3_mount(
                strategy=DockerVolumeMountStrategy(driver="rclone"),
                credentialed=True,
            )
        }
    )
    payload: dict[str, Any] = {
        "type": "test",
        "session_id": "not-a-uuid",
        "manifest": manifest.model_dump(mode="json"),
        "snapshot": NoopSnapshot(id="snapshot").model_dump(mode="json"),
    }

    with pytest.raises(ValueError):
        _SecurityTestClient().deserialize_session_state(payload)

    assert payload == {}


def test_direct_credentialless_raw_state_preserves_validation_diagnostics() -> None:
    payload: dict[str, object] = {
        "type": "test",
        "session_id": "not-a-uuid",
        "manifest": Manifest().model_dump(mode="json"),
        "snapshot": NoopSnapshot(id="snapshot").model_dump(mode="json"),
    }

    with pytest.raises(ValueError, match="session_id") as exc_info:
        TestSessionState.model_validate(payload)

    assert "UUID" in str(exc_info.value)


@pytest.mark.parametrize("boundary", ["model_validate", "client"])
@pytest.mark.parametrize("error_kind", ["base_exception", "group", "cancellation"])
def test_direct_session_state_restore_failures_are_value_free(
    boundary: str,
    error_kind: str,
) -> None:
    class ProviderAbort(BaseException):
        pass

    class ProviderCancellation(asyncio.CancelledError):
        pass

    sentinel = f"direct-{boundary}-{error_kind}-restore-secret"
    child_error: RuntimeError | None = None
    if error_kind == "cancellation":
        source_error: BaseException = ProviderCancellation(sentinel)
    elif error_kind == "group":
        child_error = RuntimeError(sentinel)
        source_error = BaseExceptionGroup(sentinel, [child_error])
    else:
        source_error = ProviderAbort(sentinel)

    class HostileMapping(Mapping[str, object]):
        def __getitem__(self, key: str) -> object:
            _ = key
            raise source_error

        def __iter__(self) -> Iterator[str]:
            raise source_error

        def __len__(self) -> int:
            return 1

    payload = HostileMapping()
    expected_error = asyncio.CancelledError if error_kind == "cancellation" else ValueError
    match = None if error_kind == "cancellation" else "sandbox session state payload is invalid"

    with pytest.raises(expected_error, match=match) as exc_info:
        if boundary == "model_validate":
            TestSessionState.model_validate(payload)
        else:
            BaseSandboxClient._deserialize_session_state_payload(payload, TestSessionState)

    assert exc_info.value is not source_error
    if error_kind == "group":
        assert source_error.args[0] == "Error details are redacted."
    else:
        assert source_error.args == ()
    assert source_error.__traceback__ is None
    if child_error is not None:
        assert child_error.args == ()
        assert child_error.__traceback__ is None
    assert sentinel not in repr(exc_info.value)
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None


@pytest.mark.parametrize(
    ("boundary", "error_kind", "expected_type", "expected_args"),
    [
        ("model_validate", "system_exit", SystemExit, (1,)),
        ("client", "keyboard_interrupt", KeyboardInterrupt, ()),
    ],
)
def test_direct_session_state_restore_preserves_value_free_process_control(
    boundary: str,
    error_kind: str,
    expected_type: type[BaseException],
    expected_args: tuple[object, ...],
) -> None:
    sentinel = f"direct-{boundary}-{error_kind}-restore-secret"
    source_error: BaseException = (
        SystemExit(sentinel) if error_kind == "system_exit" else KeyboardInterrupt(sentinel)
    )

    class HostileMapping(Mapping[str, object]):
        def __getitem__(self, key: str) -> object:
            _ = key
            raise source_error

        def __iter__(self) -> Iterator[str]:
            raise source_error

        def __len__(self) -> int:
            return 1

    payload = HostileMapping()
    with pytest.raises(expected_type) as exc_info:
        if boundary == "model_validate":
            TestSessionState.model_validate(payload)
        else:
            BaseSandboxClient._deserialize_session_state_payload(payload, TestSessionState)

    assert type(exc_info.value) is expected_type
    assert exc_info.value.args == expected_args
    assert exc_info.value is not source_error
    assert source_error.args == ()
    assert source_error.__traceback__ is None
    assert sentinel not in repr(exc_info.value)


def test_credentialless_direct_validation_preserves_value_free_process_control() -> None:
    sentinel = "direct-credentialless-validation-process-control-secret"
    source_error = SystemExit(sentinel)

    class HostilePorts:
        def __iter__(self) -> Iterator[int]:
            raise source_error

    payload: dict[str, object] = {
        "type": "test",
        "manifest": Manifest(),
        "snapshot": NoopSnapshot(id="snapshot"),
        "exposed_ports": HostilePorts(),
    }

    with pytest.raises(SystemExit) as exc_info:
        TestSessionState.model_validate(payload)

    assert type(exc_info.value) is SystemExit
    assert exc_info.value.args == (1,)
    assert exc_info.value is not source_error
    assert source_error.args == ()
    assert source_error.__traceback__ is None
    assert payload == {}
    assert sentinel not in repr(exc_info.value)


def test_credentialless_direct_json_validation_preserves_value_free_process_control() -> None:
    sentinel = "direct-credentialless-json-validation-process-control-secret"
    source_error = KeyboardInterrupt(sentinel)

    class ProcessControlSessionState(SandboxSessionState):
        type: Literal["process-control-test"] = "process-control-test"

        @field_validator("exposed_ports", mode="before")
        @classmethod
        def _raise_process_control(cls, value: object) -> object:
            _ = (cls, value)
            raise source_error

    payload = {
        "type": "process-control-test",
        "manifest": Manifest().model_dump(mode="json"),
        "snapshot": NoopSnapshot(id="snapshot").model_dump(mode="json"),
        "exposed_ports": [8080],
    }

    with pytest.raises(KeyboardInterrupt) as exc_info:
        ProcessControlSessionState.model_validate_json(json.dumps(payload))

    assert type(exc_info.value) is KeyboardInterrupt
    assert exc_info.value.args == ()
    assert exc_info.value is not source_error
    assert source_error.args == ()
    assert source_error.__traceback__ is None
    assert sentinel not in repr(exc_info.value)


def test_session_state_json_parser_preserves_value_free_process_control(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = "direct-json-parser-process-control-secret"
    source_error = KeyboardInterrupt(sentinel)

    def fail_to_parse(json_data: object) -> object:
        retained_input = json_data
        if retained_input:
            raise source_error
        return {}

    monkeypatch.setattr(json, "loads", fail_to_parse)

    with pytest.raises(KeyboardInterrupt) as exc_info:
        TestSessionState.model_validate_json(sentinel)

    assert type(exc_info.value) is KeyboardInterrupt
    assert exc_info.value.args == ()
    assert exc_info.value is not source_error
    assert source_error.args == ()
    assert source_error.__traceback__ is None
    assert sentinel not in repr(exc_info.value)


def test_session_state_pydantic_json_parser_preserves_value_free_process_control(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = "direct-pydantic-json-parser-process-control-secret"
    source_error = KeyboardInterrupt(sentinel)

    def fail_to_parse(
        cls: type[BaseModel],
        json_data: object,
        /,
        **kwargs: object,
    ) -> object:
        _ = (cls, kwargs)
        retained_input = json_data
        if retained_input:
            raise source_error
        return {}

    monkeypatch.setattr(BaseModel, "model_validate_json", classmethod(fail_to_parse))

    with pytest.raises(KeyboardInterrupt) as exc_info:
        TestSessionState.model_validate_json(json.dumps({"credential": sentinel}))

    assert type(exc_info.value) is KeyboardInterrupt
    assert exc_info.value.args == ()
    assert exc_info.value is not source_error
    assert source_error.args == ()
    assert source_error.__traceback__ is None
    assert sentinel not in repr(exc_info.value)


def test_session_state_parse_preserves_value_free_process_control_during_copy() -> None:
    sentinel = "direct-parse-process-control-secret"
    source_error = KeyboardInterrupt(sentinel)

    class HostileValue:
        def __deepcopy__(self, memo: dict[int, object]) -> object:
            _ = memo
            raise source_error

    payload: dict[str, object] = {
        "type": "test",
        "hostile": HostileValue(),
    }

    with pytest.raises(KeyboardInterrupt) as exc_info:
        SandboxSessionState.parse(payload)

    assert type(exc_info.value) is KeyboardInterrupt
    assert exc_info.value.args == ()
    assert exc_info.value is not source_error
    assert source_error.args == ()
    assert source_error.__traceback__ is None
    assert payload == {}
    assert sentinel not in repr(exc_info.value)


def test_deserialization_sanitizes_non_string_endpoint_before_validation_errors() -> None:
    sentinel = "raw-endpoint-secret"
    manifest = Manifest(
        entries={
            "data": _s3_mount(
                strategy=InContainerMountStrategy(pattern=RcloneMountPattern()),
            )
        }
    ).model_dump(mode="json")
    manifest["entries"]["data"]["endpoint_url"] = {"credential": sentinel}
    payload: dict[str, Any] = {
        "type": "test",
        "session_id": "not-a-uuid",
        "manifest": manifest,
        "snapshot": NoopSnapshot(id="snapshot").model_dump(mode="json"),
    }

    with pytest.raises(ValueError) as exc:
        _SecurityTestClient().deserialize_session_state(payload)

    assert payload == {}
    assert sentinel not in str(exc.value)


def test_serialization_failure_redacts_mount_authority_from_sdk_traceback_frames() -> None:
    sentinel = "typed-serialization-secret"
    state = TestSessionState(
        snapshot=NoopSnapshot(id="snapshot"),
        manifest=Manifest(
            entries={
                "data": S3Mount(
                    bucket="example-bucket",
                    access_key_id="access-key",
                    secret_access_key=sentinel,
                    mount_strategy=DockerVolumeMountStrategy(driver="rclone"),
                )
            }
        ),
    )
    state.snapshot = cast(Any, object())

    with pytest.raises(MountConfigError) as exc:
        _SecurityTestClient().serialize_session_state(state)

    assert sentinel not in str(exc.value)
    assert exc.value.__cause__ is None
    assert exc.value.__context__ is None
    traceback = exc.value.__traceback__
    while traceback is not None:
        module_name = traceback.tb_frame.f_globals.get("__name__", "")
        if isinstance(module_name, str) and module_name.startswith("agents."):
            assert sentinel not in repr(traceback.tb_frame.f_locals)
        traceback = traceback.tb_next


def test_direct_state_serialization_replaces_manifest_sanitizer_failure() -> None:
    sentinel = "direct-state-sanitizer-secret"
    state = TestSessionState(
        manifest=Manifest(
            entries={
                "credentials.json": LocalFile(src=Path("credentials.json")),
                "data": GCSMount(
                    bucket="bucket",
                    service_account_file="/workspace/credentials.json",
                    service_account_credentials=sentinel,
                    mount_strategy=DockerVolumeMountStrategy(driver="rclone"),
                ),
            }
        ),
        snapshot=NoopSnapshot(id="snapshot"),
    )

    with pytest.raises(Exception) as exc:
        state.model_dump(mode="json")

    assert sentinel not in str(exc.value)
    error: BaseException | None = exc.value
    while error is not None:
        traceback = error.__traceback__
        while traceback is not None:
            module_name = traceback.tb_frame.f_globals.get("__name__", "")
            if isinstance(module_name, str) and module_name.startswith("agents."):
                assert sentinel not in repr(traceback.tb_frame.f_locals)
            traceback = traceback.tb_next
        error = error.__cause__


def test_deserialization_scrubs_authority_before_invalid_strategy_discriminator() -> None:
    sentinel = "malformed-strategy-secret"
    manifest = Manifest(
        entries={
            "data": S3Mount(
                bucket="example-bucket",
                mount_strategy=DockerVolumeMountStrategy(
                    driver="rclone",
                    driver_options={"password": sentinel},
                ),
            )
        }
    ).model_dump(mode="json")
    manifest["entries"]["data"]["mount_strategy"]["type"] = {"invalid": "discriminator"}
    payload: dict[str, Any] = {
        "type": "test",
        "manifest": manifest,
        "snapshot": NoopSnapshot(id="snapshot").model_dump(mode="json"),
    }

    with pytest.raises(ValueError) as exc:
        _SecurityTestClient().deserialize_session_state(payload)

    assert payload == {}
    assert sentinel not in str(exc.value)
    traceback = exc.value.__traceback__
    while traceback is not None:
        module_name = traceback.tb_frame.f_globals.get("__name__", "")
        if isinstance(module_name, str) and module_name.startswith("agents."):
            assert sentinel not in repr(traceback.tb_frame.f_locals)
        traceback = traceback.tb_next


@pytest.mark.parametrize("location", ["strategy", "pattern"])
def test_deserialization_rejects_unknown_string_discriminators_without_values(
    location: str,
) -> None:
    sentinel = f"unknown-{location}-secret"
    manifest = Manifest(
        entries={
            "data": S3Mount(
                bucket="example-bucket",
                mount_strategy=InContainerMountStrategy(pattern=RcloneMountPattern()),
            )
        }
    ).model_dump(mode="json")
    strategy = manifest["entries"]["data"]["mount_strategy"]
    if location == "strategy":
        strategy["type"] = sentinel
    else:
        strategy["pattern"]["type"] = sentinel
    payload: dict[str, Any] = {
        "type": "test",
        "manifest": manifest,
        "snapshot": NoopSnapshot(id="snapshot").model_dump(mode="json"),
    }

    with pytest.raises(ValueError, match="payload is invalid") as exc:
        _SecurityTestClient().deserialize_session_state(payload)

    assert payload == {}
    assert sentinel not in str(exc.value)
    traceback = exc.value.__traceback__
    while traceback is not None:
        module_name = traceback.tb_frame.f_globals.get("__name__", "")
        if isinstance(module_name, str) and module_name.startswith("agents."):
            assert sentinel not in repr(traceback.tb_frame.f_locals)
        traceback = traceback.tb_next


def test_deserialization_rejects_malformed_entry_container_without_values() -> None:
    sentinel = "malformed-entry-container-secret"
    payload: dict[str, Any] = {
        "type": "test",
        "manifest": {
            "version": 1,
            "root": "/workspace",
            "entries": [sentinel],
            "environment": {"value": {}},
        },
        "snapshot": NoopSnapshot(id="snapshot").model_dump(mode="json"),
    }

    with pytest.raises(ValueError, match="payload is invalid") as exc:
        _SecurityTestClient().deserialize_session_state(payload)

    assert payload == {}
    assert sentinel not in str(exc.value)
    traceback = exc.value.__traceback__
    while traceback is not None:
        module_name = traceback.tb_frame.f_globals.get("__name__", "")
        if isinstance(module_name, str) and module_name.startswith("agents."):
            assert sentinel not in repr(traceback.tb_frame.f_locals)
        traceback = traceback.tb_next
