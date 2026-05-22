from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import Mapping
from typing import Any
from urllib.parse import parse_qs, urlparse
from urllib.request import urlopen

import pytest
import websockets

import agents.managed as managed
from agents import (
    Agent,
    ChatGPTSIWCInteractiveProvider,
    CodexHostSIWCProvider,
    CodexPluginCacheClient,
    CodexPluginServiceClient,
    LocalSIWCInteractiveProvider,
    ManagedRunner,
    SIWCAuth,
    SIWCIdentity,
    UserError,
    format_codex_plugins,
)


def _request_headers(websocket: Any) -> dict[str, str]:
    request = getattr(websocket, "request", None)
    if request is not None:
        headers = getattr(request, "headers", None)
        if headers is not None:
            return {name.lower(): value for name, value in headers.raw_items()}

    headers = getattr(websocket, "request_headers", None)
    if headers is not None:
        return {name.lower(): value for name, value in headers.raw_items()}

    return {}


@pytest.mark.asyncio
async def test_siwc_auth_from_host_reads_environment(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_SIWC_AUTHORIZATION", "biscuit:test-user")
    monkeypatch.setenv("OPENAI_CHATGPT_ACCOUNT_ID", "acct_test")

    identity = await SIWCAuth.from_host().resolve()

    assert identity.to_headers() == {
        "x-openai-authorization": "biscuit:test-user",
        "chatgpt-account-id": "acct_test",
    }


@pytest.mark.asyncio
async def test_codex_host_siwc_provider_reads_auth_file_and_mints_identity_edge_biscuit(
    tmp_path,
):
    auth_path = tmp_path / "auth.json"
    access_token = _jwt_with_chatgpt_account("acct_codex")
    auth_path.write_text(
        json.dumps({"tokens": {"access_token": access_token}}),
        encoding="utf-8",
    )
    calls: list[tuple[str, str | None, str]] = []

    async def exchange(access_token: str, account_id: str | None, env: str) -> str:
        calls.append((access_token, account_id, env))
        return "biscuit:codex-host"

    identity = await SIWCAuth.from_host(
        provider=CodexHostSIWCProvider(
            auth_path=auth_path,
            identity_edge_env="staging",
            identity_edge_exchange=exchange,
        )
    ).resolve()

    assert identity.to_headers() == {
        "x-openai-authorization": "biscuit:codex-host",
        "chatgpt-account-id": "acct_codex",
    }
    assert calls == [(access_token, "acct_codex", "staging")]


@pytest.mark.asyncio
async def test_codex_host_siwc_provider_can_mint_identity_edge_biscuit_over_http(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    auth_path = tmp_path / "auth.json"
    access_token = _jwt_with_chatgpt_account("acct_codex_http")
    auth_path.write_text(
        json.dumps({"tokens": {"access_token": access_token}}),
        encoding="utf-8",
    )
    requests: list[Any] = []

    def import_missing(_module: str) -> None:
        raise ImportError("missing")

    class Response:
        headers = {"x-openai-authorization": "biscuit:http"}

        def __enter__(self) -> Response:
            return self

        def __exit__(self, *_args: Any) -> None:
            return None

        def read(self) -> bytes:
            return b""

    def fake_urlopen(request: Any, timeout: int) -> Response:
        assert timeout == 60
        requests.append(request)
        return Response()

    monkeypatch.setattr(managed.importlib, "import_module", import_missing)
    monkeypatch.setattr(managed, "urlopen", fake_urlopen)

    identity = await SIWCAuth.from_host(
        provider=CodexHostSIWCProvider(
            auth_path=auth_path,
            identity_edge_env="staging",
        )
    ).resolve()

    assert identity.to_headers() == {
        "x-openai-authorization": "biscuit:http",
        "chatgpt-account-id": "acct_codex_http",
    }
    assert len(requests) == 1
    request = requests[0]
    assert request.full_url == (
        "https://identity-edge-namespaced-default-internal.gateway.unified-0s."
        "internal.api.openai.org/authenticate_app_v2"
    )
    assert request.headers["Authorization"] == f"Bearer {access_token}"
    assert request.headers["Chatgpt-account-id"] == "acct_codex_http"


@pytest.mark.asyncio
async def test_siwc_auth_auto_uses_host_before_interactive():
    interactive_called = False

    def interactive_provider() -> SIWCIdentity:
        nonlocal interactive_called
        interactive_called = True
        return SIWCIdentity(
            identity_edge_biscuit="biscuit:interactive",
            chatgpt_account_id="acct_interactive",
        )

    identity = await SIWCAuth.auto(
        host_provider=lambda: SIWCIdentity(
            identity_edge_biscuit="biscuit:host",
            chatgpt_account_id="acct_host",
        ),
        interactive_provider=interactive_provider,
        interactive=True,
    ).resolve()

    assert identity.identity_edge_biscuit == "biscuit:host"
    assert interactive_called is False


@pytest.mark.asyncio
async def test_siwc_auth_auto_can_fallback_to_interactive():
    identity = await SIWCAuth.auto(
        host_provider=lambda: None,
        interactive_provider=lambda: SIWCIdentity(
            identity_edge_biscuit="biscuit:interactive",
            chatgpt_account_id="acct_interactive",
        ),
        interactive=True,
    ).resolve()

    assert identity.to_headers() == {
        "x-openai-authorization": "biscuit:interactive",
        "chatgpt-account-id": "acct_interactive",
    }


@pytest.mark.asyncio
async def test_siwc_auth_interactive_errors_when_provider_returns_no_identity():
    with pytest.raises(UserError, match="interactive SIWC auth"):
        await SIWCAuth.interactive(provider=lambda: None).resolve()


@pytest.mark.asyncio
async def test_siwc_auth_interactive_local_ui_can_resume():
    provider = LocalSIWCInteractiveProvider(
        identity_edge_biscuit="biscuit:ui",
        chatgpt_account_id="acct_ui",
        open_browser=False,
    )

    task = asyncio.create_task(SIWCAuth.interactive(provider=provider).resolve())
    try:
        complete_url = await _wait_for_complete_url(provider)
        await asyncio.to_thread(lambda: urlopen(complete_url, timeout=5).read())
        identity = await task
    finally:
        if not task.done():
            task.cancel()

    assert identity.to_headers() == {
        "x-openai-authorization": "biscuit:ui",
        "chatgpt-account-id": "acct_ui",
    }


@pytest.mark.asyncio
async def test_chatgpt_siwc_interactive_provider_uses_oauth_callback_and_identity_edge_exchange():
    opened_urls: list[str] = []
    access_token = _jwt_with_chatgpt_account("acct_oauth")
    exchange_calls: list[tuple[str, str | None, str]] = []

    async def token_exchange(
        *,
        issuer: str,
        client_id: str,
        redirect_uri: str,
        code: str,
        code_verifier: str,
    ) -> dict[str, Any]:
        assert issuer == "https://auth.example.test"
        assert client_id == "app_test"
        assert code == "oauth-code"
        assert code_verifier
        assert redirect_uri.startswith("http://127.0.0.1:")
        return {"access_token": access_token}

    async def identity_edge_exchange(
        access_token: str,
        account_id: str | None,
        env: str,
    ) -> str:
        exchange_calls.append((access_token, account_id, env))
        return "biscuit:oauth"

    provider = ChatGPTSIWCInteractiveProvider(
        issuer="https://auth.example.test",
        client_id="app_test",
        callback_host="127.0.0.1",
        callback_ports=(0,),
        open_browser=lambda url: opened_urls.append(url),
        token_exchange=token_exchange,
        identity_edge_env="staging",
        identity_edge_exchange=identity_edge_exchange,
        timeout_seconds=5,
    )

    task = asyncio.create_task(provider())
    try:
        authorize_url = await _wait_for_opened_url(opened_urls)
        parsed_authorize = urlparse(authorize_url)
        query = parse_qs(parsed_authorize.query)
        redirect_uri = query["redirect_uri"][0]
        state = query["state"][0]

        await asyncio.to_thread(
            lambda: urlopen(f"{redirect_uri}?state={state}&code=oauth-code", timeout=5).read()
        )
        identity = await task
    finally:
        if not task.done():
            task.cancel()

    assert identity.to_headers() == {
        "x-openai-authorization": "biscuit:oauth",
        "chatgpt-account-id": "acct_oauth",
    }
    assert exchange_calls == [(access_token, "acct_oauth", "staging")]


@pytest.mark.asyncio
async def test_managed_runner_run_sends_siwc_headers_and_agent_run():
    captured: dict[str, Any] = {}

    async def handler(websocket: Any):
        captured["headers"] = _request_headers(websocket)
        message = await websocket.recv()
        captured["agent_run"] = json.loads(message)

        await websocket.send(
            json.dumps(
                {
                    "type": "agent.run.created",
                    "agent_run": {"id": "agrun_test", "status": "in_progress"},
                }
            )
        )
        await websocket.send(
            json.dumps(
                {
                    "type": "app_server.event",
                    "data": {
                        "method": "item/agentMessage/delta",
                        "params": {"delta": "hello "},
                    },
                }
            )
        )
        await websocket.send(
            json.dumps(
                {
                    "type": "app_server.event",
                    "data": {
                        "method": "item/agentMessage/delta",
                        "params": {"delta": "from managedagentsapi"},
                    },
                }
            )
        )
        await websocket.send(
            json.dumps(
                {
                    "type": "agent.run.idle",
                    "agent_run": {"id": "agrun_test", "status": "idle"},
                }
            )
        )

    async with websockets.serve(handler, "127.0.0.1", 0) as server:
        assert server.sockets is not None
        port = server.sockets[0].getsockname()[1]
        endpoint = f"ws://127.0.0.1:{port}/v1/agents/runs"

        result = await ManagedRunner.run(
            Agent(
                name="Managed demo",
                model="gpt-5.5",
                instructions="Use the managed runtime.",
            ),
            "Say hello.",
            auth=SIWCAuth.auto(
                host_provider=lambda: None,
                interactive_provider=lambda: SIWCIdentity(
                    identity_edge_biscuit="biscuit:interactive",
                    chatgpt_account_id="acct_interactive",
                ),
                interactive=True,
            ),
            endpoint=endpoint,
            cwd="/tmp",
        )

    assert result.final_output == "hello from managedagentsapi"
    assert captured["headers"]["x-openai-authorization"] == "biscuit:interactive"
    assert captured["headers"]["chatgpt-account-id"] == "acct_interactive"

    agent_run = captured["agent_run"]
    assert agent_run["type"] == "agent.run"
    assert agent_run["agent"]["model"] == "gpt-5.5"
    assert agent_run["agent"]["instructions"] == "Use the managed runtime."
    assert agent_run["workspace"]["manifest"]["cwd"] == "/tmp"
    assert agent_run["environment"]["type"] == "local"
    assert agent_run["input"] == [{"type": "text", "text": "Say hello."}]


@pytest.mark.asyncio
async def test_codex_plugin_service_client_lists_plugins_from_codex_chatgpt_login(
    tmp_path,
):
    auth_path = tmp_path / "auth.json"
    access_token = _jwt_with_chatgpt_account("acct_plugins")
    auth_path.write_text(
        json.dumps({"tokens": {"access_token": access_token}}),
        encoding="utf-8",
    )
    requests: list[tuple[str, dict[str, str]]] = []

    async def fetch_json(url: str, headers: Mapping[str, str]) -> dict[str, Any]:
        requests.append((url, dict(headers)))
        parsed = urlparse(url)
        query = parse_qs(parsed.query)
        key = (parsed.path, query.get("scope", [None])[0], query.get("pageToken", [None])[0])

        if key == ("/public/plugins/installed", "GLOBAL", None):
            return {
                "plugins": [
                    {
                        "id": "github@openai-curated",
                        "name": "GitHub",
                        "scope": "GLOBAL",
                        "status": "APPROVED",
                        "authentication_policy": "REQUIRED",
                        "installation_policy": "ALLOW",
                        "enabled": True,
                        "creator_name": "OpenAI",
                    }
                ],
                "pagination": {"next_page_token": "next"},
            }
        if key == ("/public/plugins/installed", "GLOBAL", "next"):
            return {
                "plugins": [
                    {
                        "id": "slack@openai-curated",
                        "name": "Slack",
                        "scope": "GLOBAL",
                        "status": "APPROVED",
                        "authentication_policy": "REQUIRED",
                        "installation_policy": "ALLOW",
                        "enabled": False,
                    }
                ],
                "pagination": {"next_page_token": None},
            }
        if key == ("/public/plugins/installed", "WORKSPACE", None):
            return {"plugins": [], "pagination": {}}
        if key == ("/public/plugins/workspace/shared", None, None):
            return {
                "plugins": [
                    {
                        "id": "team-tool",
                        "name": "Team Tool",
                        "scope": "WORKSPACE",
                        "status": "APPROVED",
                        "authentication_policy": "NONE",
                    }
                ],
                "pagination": {},
            }
        if key == ("/public/plugins/workspace/created", None, None):
            return {
                "plugins": [
                    {
                        "id": "my-plugin",
                        "name": "My Plugin",
                        "scope": "WORKSPACE",
                        "status": "DRAFT",
                        "share_url": "https://chatgpt.com/g/plugin/my-plugin",
                    }
                ],
                "pagination": {},
            }

        raise AssertionError(f"unexpected plugin request: {url}")

    client = CodexPluginServiceClient(
        plugin_service_url="https://plugins.example.test",
        auth_path=auth_path,
        fetch_json=fetch_json,
    )

    plugins = await client.list_plugins()

    assert [(plugin.id, plugin.name, plugin.source) for plugin in plugins] == [
        ("github@openai-curated", "GitHub", "installed-global"),
        ("slack@openai-curated", "Slack", "installed-global"),
        ("team-tool", "Team Tool", "shared-workspace"),
        ("my-plugin", "My Plugin", "created-workspace"),
    ]
    assert plugins[0].installed is True
    assert plugins[0].enabled is True
    assert plugins[2].installed is None
    assert plugins[3].raw["share_url"] == "https://chatgpt.com/g/plugin/my-plugin"

    assert requests
    for _, headers in requests:
        assert headers["Authorization"] == f"Bearer {access_token}"
        assert headers["ChatGPT-Account-Id"] == "acct_plugins"


@pytest.mark.asyncio
async def test_codex_plugin_service_client_uses_access_token_from_siwc_identity():
    access_token = _jwt_with_chatgpt_account("acct_from_identity")
    requested_headers: dict[str, str] = {}

    async def fetch_json(_url: str, headers: Mapping[str, str]) -> dict[str, Any]:
        requested_headers.update(headers)
        return {"plugins": [], "pagination": {}}

    identity = SIWCIdentity(
        identity_edge_biscuit="biscuit:identity",
        chatgpt_account_id="acct_from_identity",
        access_token=access_token,
    )

    client = CodexPluginServiceClient.from_siwc_identity(
        identity,
        plugin_service_url="https://plugins.example.test",
        fetch_json=fetch_json,
    )

    assert await client.list_plugins(include_workspace_installed=False) == []
    assert requested_headers["Authorization"] == f"Bearer {access_token}"
    assert requested_headers["ChatGPT-Account-Id"] == "acct_from_identity"


@pytest.mark.asyncio
async def test_codex_plugin_service_client_requires_chatgpt_login(tmp_path):
    auth_path = tmp_path / "auth.json"
    auth_path.write_text(
        json.dumps({"tokens": {"access_token": "sk-test"}}),
        encoding="utf-8",
    )
    client = CodexPluginServiceClient(
        plugin_service_url="https://plugins.example.test",
        auth_path=auth_path,
        fetch_json=lambda _url, _headers: {"plugins": [], "pagination": {}},
    )

    with pytest.raises(UserError, match="requires a ChatGPT OAuth access token"):
        await client.list_plugins()


def test_format_codex_plugins_prints_readable_summary():
    output = format_codex_plugins(
        [
            managed.CodexPlugin(
                id="github@openai-curated",
                name="GitHub",
                source="installed-global",
                installed=True,
                enabled=True,
                authentication_policy="REQUIRED",
            ),
            managed.CodexPlugin(
                id="team-tool",
                name="Team Tool",
                source="shared-workspace",
                scope="WORKSPACE",
                status="APPROVED",
            ),
        ]
    )

    assert "Codex plugins (2)" in output
    assert "- GitHub (github@openai-curated) [installed-global]" in output
    assert "installed=true enabled=true auth=REQUIRED" in output
    assert "- Team Tool (team-tool) [shared-workspace]" in output


def test_codex_plugin_cache_client_lists_enabled_plugins_from_codex_config(tmp_path):
    config_path = tmp_path / "config.toml"
    cache_dir = tmp_path / "plugins" / "cache"
    plugin_dir = cache_dir / "openai-curated" / "slack" / "004da724"
    (plugin_dir / ".codex-plugin").mkdir(parents=True)
    (plugin_dir / ".codex-plugin" / "plugin.json").write_text(
        json.dumps(
            {
                "name": "slack",
                "version": "0.1.0",
                "description": "Work with Slack.",
                "interface": {"displayName": "Slack", "developerName": "OpenAI"},
            }
        ),
        encoding="utf-8",
    )
    config_path.write_text(
        """
[plugins."slack@openai-curated"]
enabled = true

[plugins."disabled@openai-curated"]
enabled = false
""",
        encoding="utf-8",
    )

    client = CodexPluginCacheClient(config_path=config_path, plugin_cache_dir=cache_dir)

    plugins = client.list_plugins()

    assert [(plugin.id, plugin.name, plugin.source, plugin.enabled) for plugin in plugins] == [
        ("slack@openai-curated", "Slack", "codex-local-config", True)
    ]
    assert plugins[0].installed is True
    assert plugins[0].scope == "openai-curated"
    assert plugins[0].raw["version"] == "0.1.0"
    assert plugins[0].raw["path"] == str(plugin_dir)


async def _wait_for_complete_url(provider: LocalSIWCInteractiveProvider) -> str:
    for _ in range(200):
        if provider.complete_url is not None:
            return provider.complete_url
        await asyncio.sleep(0.01)
    raise AssertionError("interactive provider did not publish callback URL")


async def _wait_for_opened_url(opened_urls: list[str]) -> str:
    for _ in range(200):
        if opened_urls:
            return opened_urls[0]
        await asyncio.sleep(0.01)
    raise AssertionError("interactive provider did not open an authorize URL")


def _jwt_with_chatgpt_account(account_id: str) -> str:
    header = base64.urlsafe_b64encode(json.dumps({"alg": "none"}).encode()).rstrip(b"=")
    payload = base64.urlsafe_b64encode(
        json.dumps(
            {
                "https://api.openai.com/auth": {
                    "chatgpt_account_id": account_id,
                }
            }
        ).encode()
    ).rstrip(b"=")
    return b".".join([header, payload, b"signature"]).decode()
