from __future__ import annotations

import asyncio
import base64
import hashlib
import importlib
import inspect
import json
import os
import secrets
import threading
import uuid
import webbrowser
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Literal
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlencode, urlparse
from urllib.request import Request, urlopen

import websockets

from .agent import Agent
from .exceptions import UserError
from .models.default_models import get_default_model

IDENTITY_EDGE_BISCUIT_HEADER = "x-openai-authorization"
CHATGPT_ACCOUNT_ID_HEADER = "chatgpt-account-id"
DEFAULT_MANAGEDAGENTSAPI_ENDPOINT = "ws://127.0.0.1:8080/v1/agents/runs"
OPENAI_SIWC_AUTHORIZATION_ENV = "OPENAI_SIWC_AUTHORIZATION"
OPENAI_CHATGPT_ACCOUNT_ID_ENV = "OPENAI_CHATGPT_ACCOUNT_ID"
OPENAI_SIWC_IDENTITY_EDGE_ENV = "OPENAI_SIWC_IDENTITY_EDGE_ENV"
OPENAI_SIWC_CLIENT_ID_ENV = "OPENAI_SIWC_CLIENT_ID"
OPENAI_SIWC_ISSUER_ENV = "OPENAI_SIWC_ISSUER"
OPENAI_SIWC_CODEX_AUTH_PATH_ENV = "OPENAI_SIWC_CODEX_AUTH_PATH"
OPENAI_SIWC_IDENTITY_EDGE_URL_ENV = "OPENAI_SIWC_IDENTITY_EDGE_URL"
OPENAI_CODEX_PLUGIN_SERVICE_URL_ENV = "OPENAI_CODEX_PLUGIN_SERVICE_URL"
OPENAI_CODEX_CONFIG_PATH_ENV = "OPENAI_CODEX_CONFIG_PATH"
OPENAI_CODEX_PLUGIN_CACHE_DIR_ENV = "OPENAI_CODEX_PLUGIN_CACHE_DIR"
CODEX_AUTH_PATH_ENV = "CODEX_AUTH_PATH"
CODEX_HOME_ENV = "CODEX_HOME"
CODEX_PLUGIN_ACCOUNT_ID_HEADER = "ChatGPT-Account-Id"
DEFAULT_CODEX_PLUGIN_PAGE_LIMIT = 100
DEFAULT_CHATGPT_SIWC_ISSUER = "https://auth.openai.com"
DEFAULT_CHATGPT_SIWC_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
DEFAULT_CHATGPT_SIWC_SCOPES = (
    "openid",
    "email",
    "profile",
    "offline_access",
    "api.connectors.read",
    "api.connectors.invoke",
)

SIWCIdentityProvider = Callable[[], "SIWCIdentity | None | Awaitable[SIWCIdentity | None]"]
SIWCIdentityEdgeExchange = Callable[[str, str | None, str], "str | Awaitable[str]"]
SIWCTokenExchange = Callable[..., "Mapping[str, Any] | Awaitable[Mapping[str, Any]]"]
BrowserOpener = Callable[[str], "object | Awaitable[object]"]
CodexPluginFetchJSON = Callable[
    [str, Mapping[str, str]], "Mapping[str, Any] | Awaitable[Mapping[str, Any]]"
]


@dataclass(frozen=True)
class SIWCIdentity:
    """Identity-edge credentials for Sign in with ChatGPT managed agent runs."""

    identity_edge_biscuit: str
    chatgpt_account_id: str
    access_token: str | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.identity_edge_biscuit.strip():
            raise UserError("SIWC identity_edge_biscuit must be non-empty.")
        if not self.chatgpt_account_id.strip():
            raise UserError("SIWC chatgpt_account_id must be non-empty.")
        if self.access_token is not None and not self.access_token.strip():
            raise UserError("SIWC access_token must be non-empty when provided.")

    def to_headers(self) -> dict[str, str]:
        return {
            IDENTITY_EDGE_BISCUIT_HEADER: self.identity_edge_biscuit,
            CHATGPT_ACCOUNT_ID_HEADER: self.chatgpt_account_id,
        }


@dataclass(frozen=True)
class SIWCAuth:
    """Resolver for Sign in with ChatGPT identity used by ManagedRunner."""

    _mode: Literal["host", "interactive", "auto"]
    _host_provider: SIWCIdentityProvider | None = None
    _interactive_provider: SIWCIdentityProvider | None = None
    _allow_interactive: bool = False

    @classmethod
    def from_host(cls, provider: SIWCIdentityProvider | None = None) -> SIWCAuth:
        """Use identity from the embedding host without opening interactive UI."""
        return cls(_mode="host", _host_provider=provider or CodexHostSIWCProvider())

    @classmethod
    def interactive(cls, provider: SIWCIdentityProvider | None = None) -> SIWCAuth:
        """Use an explicit interactive sign-in provider supplied by the host or app."""
        return cls(
            _mode="interactive",
            _interactive_provider=provider or ChatGPTSIWCInteractiveProvider(),
        )

    @classmethod
    def auto(
        cls,
        *,
        host_provider: SIWCIdentityProvider | None = None,
        interactive_provider: SIWCIdentityProvider | None = None,
        interactive: bool = False,
    ) -> SIWCAuth:
        """Try host identity first, then optionally fall back to interactive sign-in."""
        return cls(
            _mode="auto",
            _host_provider=host_provider or CodexHostSIWCProvider(),
            _interactive_provider=interactive_provider
            or (ChatGPTSIWCInteractiveProvider() if interactive else None),
            _allow_interactive=interactive,
        )

    async def resolve(self) -> SIWCIdentity:
        if self._mode == "host":
            identity = await _call_identity_provider(self._host_provider)
            if identity is None:
                raise UserError(
                    "SIWC host identity is unavailable. Run inside a ChatGPT/Codex host or "
                    f"set {OPENAI_SIWC_AUTHORIZATION_ENV} and {OPENAI_CHATGPT_ACCOUNT_ID_ENV}."
                )
            return identity

        if self._mode == "interactive":
            identity = await _call_identity_provider(self._interactive_provider)
            if identity is None:
                raise UserError(
                    "interactive SIWC auth requires an interactive provider supplied by the "
                    "host or application."
                )
            return identity

        identity = await _call_identity_provider(self._host_provider)
        if identity is not None:
            return identity

        if not self._allow_interactive:
            raise UserError(
                "SIWC host identity is unavailable and interactive fallback is disabled."
            )

        identity = await _call_identity_provider(self._interactive_provider)
        if identity is None:
            raise UserError(
                "interactive SIWC auth requires an interactive provider supplied by the "
                "host or application."
            )
        return identity


@dataclass
class CodexHostSIWCProvider:
    """Resolve real SIWC identity from an existing Codex ChatGPT login.

    The host path first honors explicit SIWC identity headers from the environment. If
    those are not set, it reads the Codex auth file, extracts the ChatGPT user token
    and account id, and exchanges the token through Identity Edge for the actor biscuit
    expected by managedagentsapi/cloud-router.
    """

    auth_path: str | Path | None = None
    identity_edge_env: str | None = None
    identity_edge_exchange: SIWCIdentityEdgeExchange = field(
        default_factory=lambda: _mint_identity_edge_biscuit
    )

    async def __call__(self) -> SIWCIdentity | None:
        env_identity = _siwc_identity_from_environment()
        if env_identity is not None:
            return env_identity

        access_token, account_id = _codex_auth_file_tokens(self.auth_path)
        if access_token is None:
            return None
        if access_token.startswith(("sk-", "sess-", "pk-")):
            return None
        if account_id is None:
            raise UserError(
                "Codex ChatGPT auth is present, but the ChatGPT account id is missing. "
                "Run ChatGPT sign-in again or pass an explicit SIWC provider."
            )

        biscuit = await _call_identity_edge_exchange(
            self.identity_edge_exchange,
            access_token,
            account_id,
            self.identity_edge_env or _identity_edge_env_from_environment(),
        )
        return SIWCIdentity(
            identity_edge_biscuit=biscuit,
            chatgpt_account_id=account_id,
            access_token=access_token,
        )


@dataclass
class ChatGPTSIWCInteractiveProvider:
    """Run a real browser Sign in with ChatGPT flow and mint an Identity Edge biscuit.

    This provider uses AuthAPI's OAuth authorization-code + PKCE flow with the Codex
    client by default. After receiving the OpenAI access token, it exchanges that
    token through Identity Edge so `ManagedRunner` can call managedagentsapi using
    the same biscuit-backed identity shape as Codex/cloud-router.
    """

    issuer: str | None = None
    client_id: str | None = None
    scopes: Sequence[str] = DEFAULT_CHATGPT_SIWC_SCOPES
    callback_host: str = "localhost"
    callback_ports: Sequence[int] = (1455, 1457)
    callback_path: str = "/auth/callback"
    open_browser: BrowserOpener | None = None
    token_exchange: SIWCTokenExchange = field(
        default_factory=lambda: _oauth_authorization_code_token_exchange
    )
    identity_edge_env: str | None = None
    identity_edge_exchange: SIWCIdentityEdgeExchange = field(
        default_factory=lambda: _mint_identity_edge_biscuit
    )
    timeout_seconds: float = 600
    extra_authorize_params: Mapping[str, str] | None = None

    async def __call__(self) -> SIWCIdentity:
        issuer = _issuer_without_trailing_slash(
            self.issuer or os.getenv(OPENAI_SIWC_ISSUER_ENV) or DEFAULT_CHATGPT_SIWC_ISSUER
        )
        client_id = (
            self.client_id or os.getenv(OPENAI_SIWC_CLIENT_ID_ENV) or DEFAULT_CHATGPT_SIWC_CLIENT_ID
        )
        pkce = _create_pkce()
        state = secrets.token_urlsafe(32)
        callback = _OAuthCallbackServer(
            host=self.callback_host,
            ports=self.callback_ports,
            path=self.callback_path,
            state=state,
        )
        callback.start()

        try:
            authorize_url = _build_chatgpt_siwc_authorize_url(
                issuer=issuer,
                client_id=client_id,
                redirect_uri=callback.redirect_uri,
                scopes=self.scopes,
                code_challenge=pkce["code_challenge"],
                state=state,
                extra_params=self.extra_authorize_params,
            )
            await _maybe_await((self.open_browser or webbrowser.open)(authorize_url))
            code = await asyncio.wait_for(callback.authorization_code, self.timeout_seconds)
            token_response = await _call_token_exchange(
                self.token_exchange,
                issuer=issuer,
                client_id=client_id,
                redirect_uri=callback.redirect_uri,
                code=code,
                code_verifier=pkce["code_verifier"],
            )
            access_token = _non_empty_string(token_response.get("access_token"))
            if access_token is None:
                raise UserError("SIWC token exchange did not return an OpenAI access token.")
            account_id = _non_empty_string(
                token_response.get("account_id")
            ) or _extract_chatgpt_account_id(access_token)
            if account_id is None:
                raise UserError(
                    "SIWC access token does not include a ChatGPT account id. "
                    "Select a ChatGPT workspace and retry sign-in."
                )

            biscuit = await _call_identity_edge_exchange(
                self.identity_edge_exchange,
                access_token,
                account_id,
                self.identity_edge_env or _identity_edge_env_from_environment(),
            )
            return SIWCIdentity(
                identity_edge_biscuit=biscuit,
                chatgpt_account_id=account_id,
                access_token=access_token,
            )
        finally:
            await callback.close()


@dataclass(frozen=True)
class ManagedDynamicTool:
    """A model-visible dynamic tool declaration for managedagentsapi."""

    name: str
    input_schema: Mapping[str, Any]
    description: str | None = None
    namespace: str | None = None
    defer_loading: bool | None = None

    def to_agent_run_json(self) -> dict[str, Any]:
        if not self.name.strip():
            raise UserError("ManagedDynamicTool.name must be non-empty.")
        if not isinstance(self.input_schema, Mapping):
            raise UserError("ManagedDynamicTool.input_schema must be a JSON schema object.")

        tool: dict[str, Any] = {
            "name": self.name,
            "inputSchema": dict(self.input_schema),
        }
        if self.description is not None:
            tool["description"] = self.description
        if self.namespace is not None:
            tool["namespace"] = self.namespace
        if self.defer_loading is not None:
            tool["deferLoading"] = self.defer_loading
        return tool


@dataclass(frozen=True)
class CodexPlugin:
    """A Codex plugin visible to the signed-in ChatGPT identity."""

    id: str
    name: str
    source: str
    scope: str | None = None
    installed: bool | None = None
    enabled: bool | None = None
    status: str | None = None
    authentication_policy: str | None = None
    installation_policy: str | None = None
    creator_name: str | None = None
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise UserError("CodexPlugin.id must be non-empty.")
        if not self.name.strip():
            raise UserError("CodexPlugin.name must be non-empty.")
        if not self.source.strip():
            raise UserError("CodexPlugin.source must be non-empty.")


@dataclass
class CodexPluginServiceClient:
    """List Codex plugins available to a ChatGPT/Codex identity.

    The client uses the ChatGPT OAuth access token from an existing Codex login, or
    from a `SIWCIdentity` returned by an interactive SIWC flow. It intentionally
    returns normalized plugin metadata only; tool loading/invocation remains the next
    layer on top of managedagentsapi.
    """

    plugin_service_url: str | None = None
    auth_path: str | Path | None = None
    access_token: str | None = field(default=None, repr=False)
    chatgpt_account_id: str | None = None
    fetch_json: CodexPluginFetchJSON = field(
        default_factory=lambda: _codex_plugin_service_fetch_json
    )

    @classmethod
    def from_siwc_identity(
        cls,
        identity: SIWCIdentity,
        *,
        plugin_service_url: str | None = None,
        fetch_json: CodexPluginFetchJSON | None = None,
    ) -> CodexPluginServiceClient:
        kwargs: dict[str, Any] = {
            "plugin_service_url": plugin_service_url,
            "access_token": identity.access_token,
            "chatgpt_account_id": identity.chatgpt_account_id,
        }
        if fetch_json is not None:
            kwargs["fetch_json"] = fetch_json
        return cls(**kwargs)

    async def list_plugins(
        self,
        *,
        include_global_installed: bool = True,
        include_workspace_installed: bool = True,
        include_shared_workspace: bool = True,
        include_created_workspace: bool = True,
        include_available: bool = False,
        limit: int = DEFAULT_CODEX_PLUGIN_PAGE_LIMIT,
    ) -> list[CodexPlugin]:
        headers = self._auth_headers()
        plugins: list[CodexPlugin] = []

        if include_global_installed:
            plugins.extend(
                await self._list_plugin_collection(
                    "/public/plugins/installed",
                    params={"scope": "GLOBAL"},
                    source="installed-global",
                    installed=True,
                    headers=headers,
                    limit=limit,
                )
            )
        if include_workspace_installed:
            plugins.extend(
                await self._list_plugin_collection(
                    "/public/plugins/installed",
                    params={"scope": "WORKSPACE"},
                    source="installed-workspace",
                    installed=True,
                    headers=headers,
                    limit=limit,
                )
            )
        if include_shared_workspace:
            plugins.extend(
                await self._list_plugin_collection(
                    "/public/plugins/workspace/shared",
                    params={},
                    source="shared-workspace",
                    installed=None,
                    headers=headers,
                    limit=limit,
                )
            )
        if include_created_workspace:
            plugins.extend(
                await self._list_plugin_collection(
                    "/public/plugins/workspace/created",
                    params={},
                    source="created-workspace",
                    installed=None,
                    headers=headers,
                    limit=limit,
                )
            )
        if include_available:
            for scope, source in [
                ("GLOBAL", "available-global"),
                ("WORKSPACE", "available-workspace"),
            ]:
                plugins.extend(
                    await self._list_plugin_collection(
                        "/public/plugins/list",
                        params={"scope": scope},
                        source=source,
                        installed=False,
                        headers=headers,
                        limit=limit,
                    )
                )

        return plugins

    async def _list_plugin_collection(
        self,
        path: str,
        *,
        params: Mapping[str, object],
        source: str,
        installed: bool | None,
        headers: Mapping[str, str],
        limit: int,
    ) -> list[CodexPlugin]:
        page_token: str | None = None
        plugins: list[CodexPlugin] = []

        while True:
            request_params: dict[str, object] = {
                "limit": limit,
                "includeDownloadUrls": False,
                **params,
            }
            if page_token is not None:
                request_params["pageToken"] = page_token
            payload = await self._fetch_json(self._url(path, request_params), headers)
            items = payload.get("plugins")
            if not isinstance(items, Sequence) or isinstance(items, str | bytes):
                raise UserError(f"Plugin Service response for {path} did not include plugins.")

            for item in items:
                if not isinstance(item, Mapping):
                    raise UserError(f"Plugin Service response for {path} included malformed item.")
                plugins.append(_codex_plugin_from_payload(item, source, installed))

            pagination = payload.get("pagination")
            page_token = None
            if isinstance(pagination, Mapping):
                page_token = _non_empty_string(
                    pagination.get("next_page_token") or pagination.get("nextPageToken")
                )
            if page_token is None:
                return plugins

    async def _fetch_json(
        self,
        url: str,
        headers: Mapping[str, str],
    ) -> Mapping[str, Any]:
        payload = await _maybe_await(self.fetch_json(url, headers))
        if not isinstance(payload, Mapping):
            raise UserError("Plugin Service returned a malformed JSON response.")
        return payload

    def _url(self, path: str, params: Mapping[str, object]) -> str:
        base_url = _codex_plugin_service_base_url(self.plugin_service_url)
        query = urlencode(
            {key: _query_param_value(value) for key, value in params.items() if value is not None}
        )
        if base_url.endswith("/public") and path.startswith("/public/"):
            path = path[len("/public") :]
        return f"{base_url}{path}?{query}" if query else f"{base_url}{path}"

    def _auth_headers(self) -> dict[str, str]:
        access_token = _non_empty_string(self.access_token)
        account_id = _non_empty_string(self.chatgpt_account_id)
        if access_token is None:
            access_token, file_account_id = _codex_auth_file_tokens(self.auth_path)
            account_id = account_id or file_account_id

        if access_token is None:
            raise UserError(
                "Codex plugin listing requires a ChatGPT login. Run Codex ChatGPT sign-in "
                "or pass a SIWC identity that includes an access token."
            )
        if access_token.startswith(("sk-", "sess-", "pk-")):
            raise UserError(
                "Codex plugin listing requires a ChatGPT OAuth access token, not an API "
                "key or session token."
            )
        account_id = account_id or _extract_chatgpt_account_id(access_token)
        if account_id is None:
            raise UserError(
                "Codex plugin listing requires a ChatGPT account id. Select a ChatGPT "
                "workspace and sign in again."
            )
        return {
            "Authorization": f"Bearer {access_token}",
            CODEX_PLUGIN_ACCOUNT_ID_HEADER: account_id,
        }


@dataclass
class CodexPluginCacheClient:
    """List plugins enabled in the local Codex config and cached plugin packages."""

    config_path: str | Path | None = None
    plugin_cache_dir: str | Path | None = None

    def list_plugins(self, *, include_disabled: bool = False) -> list[CodexPlugin]:
        config = _read_codex_plugin_config(_codex_config_path(self.config_path))
        cache_dir = _codex_plugin_cache_dir(self.plugin_cache_dir)
        plugins: list[CodexPlugin] = []

        for plugin_id, plugin_config in config.items():
            enabled = _optional_bool(plugin_config.get("enabled"))
            if enabled is False and not include_disabled:
                continue
            plugins.append(_codex_plugin_from_cache(plugin_id, plugin_config, cache_dir))

        return plugins


def format_codex_plugins(plugins: Sequence[CodexPlugin]) -> str:
    """Format Codex plugins for CLI/demo output."""

    rows = list(plugins)
    lines = [f"Codex plugins ({len(rows)})"]
    if not rows:
        lines.append("No Codex plugins found for this ChatGPT identity.")
        return "\n".join(lines)

    for plugin in rows:
        fields: list[str] = []
        if plugin.installed is not None:
            fields.append(f"installed={_bool_label(plugin.installed)}")
        if plugin.enabled is not None:
            fields.append(f"enabled={_bool_label(plugin.enabled)}")
        if plugin.authentication_policy is not None:
            fields.append(f"auth={plugin.authentication_policy}")
        if plugin.installation_policy is not None:
            fields.append(f"install={plugin.installation_policy}")
        if plugin.scope is not None:
            fields.append(f"scope={plugin.scope}")
        if plugin.status is not None:
            fields.append(f"status={plugin.status}")
        if plugin.creator_name is not None:
            fields.append(f"creator={plugin.creator_name}")

        line = f"- {plugin.name} ({plugin.id}) [{plugin.source}]"
        if fields:
            line = f"{line} {' '.join(fields)}"
        lines.append(line)
    return "\n".join(lines)


@dataclass
class LocalSIWCInteractiveProvider:
    """A loopback demo UI for interactive Sign in with ChatGPT flows.

    This provider is intended for local demos. It does not talk to ChatGPT or mint a real
    identity-edge biscuit; instead it waits for the local page to be completed and returns
    the configured demo identity.
    """

    identity_edge_biscuit: str = "biscuit:local-demo-user"
    chatgpt_account_id: str = "acct_local_demo"
    host: str = "127.0.0.1"
    port: int = 0
    open_browser: bool = True
    timeout_seconds: float = 120
    authorize_url: str | None = field(default=None, init=False)
    complete_url: str | None = field(default=None, init=False)

    async def __call__(self) -> SIWCIdentity:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[SIWCIdentity] = loop.create_future()
        state = uuid.uuid4().hex
        provider = self

        def resolve(identity: SIWCIdentity) -> None:
            if not future.done():
                future.set_result(identity)

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                parsed = urlparse(self.path)
                if parsed.path == "/":
                    self._send_html(HTTPStatus.OK, provider._sign_in_page(state))
                    return

                if parsed.path == "/complete":
                    query = parse_qs(parsed.query)
                    if query.get("state") != [state]:
                        self._send_html(
                            HTTPStatus.FORBIDDEN,
                            "<h1>Invalid sign-in state</h1>",
                        )
                        return

                    identity = SIWCIdentity(
                        identity_edge_biscuit=provider.identity_edge_biscuit,
                        chatgpt_account_id=provider.chatgpt_account_id,
                    )
                    loop.call_soon_threadsafe(resolve, identity)
                    self._send_html(HTTPStatus.OK, provider._success_page())
                    return

                self._send_html(HTTPStatus.NOT_FOUND, "<h1>Not found</h1>")

            def log_message(self, format: str, *args: Any) -> None:
                return None

            def _send_html(self, status: HTTPStatus, body: str) -> None:
                encoded = body.encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

        server = ThreadingHTTPServer((self.host, self.port), Handler)
        actual_port = server.server_address[1]
        self.authorize_url = f"http://{self.host}:{actual_port}/?state={state}"
        self.complete_url = f"http://{self.host}:{actual_port}/complete?state={state}"
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        try:
            if self.open_browser:
                webbrowser.open(self.authorize_url)
            return await asyncio.wait_for(future, timeout=self.timeout_seconds)
        finally:
            await asyncio.to_thread(server.shutdown)
            server.server_close()
            thread.join(timeout=1)

    def _sign_in_page(self, state: str) -> str:
        return f"""<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>Sign in with ChatGPT</title>
    <style>
      body {{
        font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
          "Segoe UI", sans-serif;
        margin: 0;
        min-height: 100vh;
        display: grid;
        place-items: center;
        background: #f7f7f4;
        color: #111;
      }}
      main {{
        width: min(440px, calc(100vw - 32px));
        border: 1px solid #d9d9d0;
        border-radius: 8px;
        padding: 28px;
        background: white;
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.08);
      }}
      h1 {{
        margin: 0 0 10px;
        font-size: 24px;
      }}
      p {{
        line-height: 1.5;
        color: #444;
      }}
      button {{
        width: 100%;
        border: 0;
        border-radius: 6px;
        padding: 12px 16px;
        font-size: 15px;
        font-weight: 600;
        background: #111;
        color: white;
        cursor: pointer;
      }}
      code {{
        overflow-wrap: anywhere;
      }}
    </style>
  </head>
  <body>
    <main>
      <h1>Sign in with ChatGPT</h1>
      <p>
        This is a local demo sign-in page. It simulates host-provided SIWC identity so the
        Agents SDK managed run can continue.
      </p>
      <p>Demo account: <code>{self.chatgpt_account_id}</code></p>
      <form method="get" action="/complete">
        <input type="hidden" name="state" value="{state}" />
        <button type="submit">Continue with ChatGPT</button>
      </form>
    </main>
  </body>
</html>"""

    def _success_page(self) -> str:
        return """<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>Signed in</title>
  </head>
  <body style="font-family: ui-sans-serif, system-ui; padding: 32px;">
    <h1>Signed in with ChatGPT</h1>
    <p>The local demo identity was sent back to the Agents SDK. You can close this tab.</p>
  </body>
</html>"""


@dataclass
class ManagedRunResult:
    """Result returned by ManagedRunner.run."""

    final_output: str
    events: list[dict[str, Any]]
    agent_run: dict[str, Any] | None = None
    raw_agent_run_request: dict[str, Any] | None = None


class ManagedRunner:
    """Run an Agent through managedagentsapi using SIWC identity."""

    @classmethod
    async def run(
        cls,
        starting_agent: Agent[Any],
        input: str | Sequence[Mapping[str, Any]],
        *,
        auth: SIWCAuth,
        endpoint: str = DEFAULT_MANAGEDAGENTSAPI_ENDPOINT,
        cwd: str | None = None,
        environment: Literal["local", "delegated"] = "local",
        dynamic_tools: Sequence[ManagedDynamicTool | Mapping[str, Any]] | None = None,
        extra_headers: Mapping[str, str] | None = None,
    ) -> ManagedRunResult:
        identity = await auth.resolve()
        headers = identity.to_headers()
        if extra_headers:
            headers.update(extra_headers)

        request = _agent_run_request(
            starting_agent,
            input,
            cwd=cwd,
            environment=environment,
            dynamic_tools=dynamic_tools,
        )

        events: list[dict[str, Any]] = []
        output_chunks: list[str] = []
        agent_run: dict[str, Any] | None = None

        async with websockets.connect(endpoint, **_connect_header_kwargs(headers)) as websocket:
            await websocket.send(json.dumps(request))
            async for message in websocket:
                event = json.loads(message)
                events.append(event)
                if event.get("type") == "error":
                    message = (
                        event.get("error", {}).get("message")
                        if isinstance(event.get("error"), Mapping)
                        else None
                    )
                    raise UserError(message or f"managedagentsapi returned error: {event}")

                if isinstance(event.get("agent_run"), Mapping):
                    agent_run = dict(event["agent_run"])

                delta = _agent_message_delta(event)
                if delta is not None:
                    output_chunks.append(delta)

                if event.get("type") in {"agent.run.idle", "agent.run.completed"}:
                    break

        return ManagedRunResult(
            final_output="".join(output_chunks),
            events=events,
            agent_run=agent_run,
            raw_agent_run_request=request,
        )


async def _call_identity_provider(provider: SIWCIdentityProvider | None) -> SIWCIdentity | None:
    if provider is None:
        return None
    result = provider()
    if inspect.isawaitable(result):
        result = await result
    return result


async def _call_identity_edge_exchange(
    exchange: SIWCIdentityEdgeExchange,
    access_token: str,
    account_id: str | None,
    env: str,
) -> str:
    result = exchange(access_token, account_id, env)
    if inspect.isawaitable(result):
        result = await result
    if not result.strip():
        raise UserError("SIWC Identity Edge exchange returned an empty biscuit.")
    return result


async def _call_token_exchange(exchange: SIWCTokenExchange, **kwargs: Any) -> Mapping[str, Any]:
    result = exchange(**kwargs)
    if inspect.isawaitable(result):
        result = await result
    return result


def _siwc_identity_from_environment() -> SIWCIdentity | None:
    biscuit = os.getenv(OPENAI_SIWC_AUTHORIZATION_ENV)
    account_id = os.getenv(OPENAI_CHATGPT_ACCOUNT_ID_ENV)
    if not biscuit or not account_id:
        return None
    return SIWCIdentity(identity_edge_biscuit=biscuit, chatgpt_account_id=account_id)


def _identity_edge_env_from_environment() -> str:
    return os.getenv(OPENAI_SIWC_IDENTITY_EDGE_ENV, "prod")


def _codex_auth_file_tokens(auth_path: str | Path | None) -> tuple[str | None, str | None]:
    path = _codex_auth_path(auth_path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, None
    if not isinstance(payload, Mapping):
        return None, None
    tokens = payload.get("tokens")
    if not isinstance(tokens, Mapping):
        return None, None
    access_token = _non_empty_string(tokens.get("access_token"))
    account_id = _non_empty_string(tokens.get("account_id"))
    if account_id is None and access_token is not None:
        account_id = _extract_chatgpt_account_id(access_token)
    return access_token, account_id


def _codex_auth_path(auth_path: str | Path | None) -> Path:
    if auth_path is not None:
        return Path(auth_path).expanduser()
    if os.environ.get(OPENAI_SIWC_CODEX_AUTH_PATH_ENV):
        return Path(os.environ[OPENAI_SIWC_CODEX_AUTH_PATH_ENV]).expanduser()
    if os.environ.get(CODEX_AUTH_PATH_ENV):
        return Path(os.environ[CODEX_AUTH_PATH_ENV]).expanduser()
    if os.environ.get(CODEX_HOME_ENV):
        return Path(os.environ[CODEX_HOME_ENV]).expanduser() / "auth.json"
    return Path.home() / ".codex" / "auth.json"


def _codex_config_path(config_path: str | Path | None) -> Path:
    if config_path is not None:
        return Path(config_path).expanduser()
    if os.environ.get(OPENAI_CODEX_CONFIG_PATH_ENV):
        return Path(os.environ[OPENAI_CODEX_CONFIG_PATH_ENV]).expanduser()
    if os.environ.get(CODEX_HOME_ENV):
        return Path(os.environ[CODEX_HOME_ENV]).expanduser() / "config.toml"
    return Path.home() / ".codex" / "config.toml"


def _codex_plugin_cache_dir(plugin_cache_dir: str | Path | None) -> Path:
    if plugin_cache_dir is not None:
        return Path(plugin_cache_dir).expanduser()
    if os.environ.get(OPENAI_CODEX_PLUGIN_CACHE_DIR_ENV):
        return Path(os.environ[OPENAI_CODEX_PLUGIN_CACHE_DIR_ENV]).expanduser()
    if os.environ.get(CODEX_HOME_ENV):
        return Path(os.environ[CODEX_HOME_ENV]).expanduser() / "plugins" / "cache"
    return Path.home() / ".codex" / "plugins" / "cache"


def _read_codex_plugin_config(config_path: Path) -> dict[str, dict[str, Any]]:
    try:
        lines = config_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}

    plugins: dict[str, dict[str, Any]] = {}
    current_plugin_id: str | None = None
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[plugins.") and line.endswith("]"):
            current_plugin_id = _parse_codex_plugin_table_name(line)
            if current_plugin_id is not None:
                plugins.setdefault(current_plugin_id, {})
            continue
        if current_plugin_id is None or "=" not in line:
            continue

        key, value = [part.strip() for part in line.split("=", 1)]
        plugins[current_plugin_id][key] = _parse_codex_config_value(value)
    return plugins


def _parse_codex_plugin_table_name(line: str) -> str | None:
    raw_name = line[len("[plugins.") : -1].strip()
    if not raw_name:
        return None
    if raw_name.startswith('"') and raw_name.endswith('"'):
        try:
            value = json.loads(raw_name)
        except json.JSONDecodeError:
            return None
        return _non_empty_string(value)
    return _non_empty_string(raw_name)


def _parse_codex_config_value(value: str) -> object:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if value.startswith('"') and value.endswith('"'):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value[1:-1]
    return value


def _extract_chatgpt_account_id(access_token: str) -> str | None:
    try:
        encoded_payload = access_token.split(".")[1]
        encoded_payload += "=" * (-len(encoded_payload) % 4)
        payload = json.loads(base64.urlsafe_b64decode(encoded_payload))
        auth_claims = payload["https://api.openai.com/auth"]
        account_id = auth_claims["chatgpt_account_id"]
    except (IndexError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    return _non_empty_string(account_id)


def _non_empty_string(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _codex_plugin_service_base_url(plugin_service_url: str | None) -> str:
    base_url = _non_empty_string(plugin_service_url) or _non_empty_string(
        os.getenv(OPENAI_CODEX_PLUGIN_SERVICE_URL_ENV)
    )
    if base_url is None:
        raise UserError(
            "Codex plugin listing requires a Plugin Service URL. Pass plugin_service_url "
            f"or set {OPENAI_CODEX_PLUGIN_SERVICE_URL_ENV}."
        )
    return base_url.rstrip("/")


def _query_param_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _codex_plugin_from_payload(
    payload: Mapping[str, Any],
    source: str,
    installed: bool | None,
) -> CodexPlugin:
    plugin_id = _first_non_empty_string(payload, "id", "plugin_id", "pluginId")
    if plugin_id is None:
        raise UserError("Plugin Service response included a plugin without an id.")
    name = _first_non_empty_string(payload, "name", "display_name", "displayName") or plugin_id
    return CodexPlugin(
        id=plugin_id,
        name=name,
        source=source,
        scope=_first_non_empty_string(payload, "scope"),
        installed=installed if installed is not None else _optional_bool(payload.get("installed")),
        enabled=_optional_bool(payload.get("enabled")),
        status=_first_non_empty_string(payload, "status"),
        authentication_policy=_first_non_empty_string(
            payload,
            "authentication_policy",
            "authenticationPolicy",
            "auth_policy",
            "authPolicy",
        ),
        installation_policy=_first_non_empty_string(
            payload,
            "installation_policy",
            "installationPolicy",
            "install_policy",
            "installPolicy",
        ),
        creator_name=_first_non_empty_string(payload, "creator_name", "creatorName"),
        raw=dict(payload),
    )


def _codex_plugin_from_cache(
    plugin_id: str,
    plugin_config: Mapping[str, Any],
    cache_dir: Path,
) -> CodexPlugin:
    plugin_name, marketplace = _split_codex_plugin_id(plugin_id)
    plugin_path, manifest = _read_codex_cached_plugin_manifest(
        cache_dir=cache_dir,
        plugin_name=plugin_name,
        marketplace=marketplace,
    )
    interface = manifest.get("interface")
    if not isinstance(interface, Mapping):
        interface = {}

    display_name = _first_non_empty_string(interface, "displayName", "display_name")
    manifest_name = _first_non_empty_string(manifest, "name")
    enabled = _optional_bool(plugin_config.get("enabled"))
    raw = dict(manifest)
    raw["id"] = plugin_id
    raw["marketplace"] = marketplace
    raw["config"] = dict(plugin_config)
    if plugin_path is not None:
        raw["path"] = str(plugin_path)

    return CodexPlugin(
        id=plugin_id,
        name=display_name or manifest_name or plugin_name,
        source="codex-local-config",
        scope=marketplace,
        installed=True,
        enabled=enabled,
        status="enabled" if enabled is True else "disabled" if enabled is False else None,
        creator_name=_first_non_empty_string(interface, "developerName", "developer_name")
        or _author_name(manifest.get("author")),
        raw=raw,
    )


def _split_codex_plugin_id(plugin_id: str) -> tuple[str, str | None]:
    plugin_name, separator, marketplace = plugin_id.partition("@")
    return plugin_name, marketplace if separator else None


def _read_codex_cached_plugin_manifest(
    *,
    cache_dir: Path,
    plugin_name: str,
    marketplace: str | None,
) -> tuple[Path | None, Mapping[str, Any]]:
    search_roots = [cache_dir / marketplace / plugin_name] if marketplace else []
    search_roots.append(cache_dir / plugin_name)

    for root in search_roots:
        if not root.exists():
            continue
        for version_dir in sorted(root.iterdir(), key=lambda path: path.name, reverse=True):
            manifest_path = version_dir / ".codex-plugin" / "plugin.json"
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(manifest, Mapping):
                return version_dir, manifest

    return None, {}


def _author_name(author: object) -> str | None:
    if isinstance(author, Mapping):
        return _non_empty_string(author.get("name"))
    return _non_empty_string(author)


def _first_non_empty_string(payload: Mapping[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = _non_empty_string(payload.get(key))
        if value is not None:
            return value
    return None


def _optional_bool(value: object) -> bool | None:
    return value if isinstance(value, bool) else None


def _bool_label(value: bool) -> str:
    return "true" if value else "false"


def _codex_plugin_service_fetch_json(
    url: str,
    headers: Mapping[str, str],
) -> Mapping[str, Any]:
    request = Request(url, headers=dict(headers), method="GET")
    try:
        with urlopen(request, timeout=60) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:500]
        raise UserError(
            f"Plugin Service request failed with HTTP {exc.code}: {body or exc.reason}"
        ) from exc
    except (URLError, OSError) as exc:
        raise UserError(
            f"Plugin Service request failed: {exc}. Start or port-forward Plugin Service, "
            "or use the local Codex plugin cache."
        ) from exc
    except json.JSONDecodeError as exc:
        raise UserError("Plugin Service returned invalid JSON.") from exc

    if not isinstance(payload, Mapping):
        raise UserError("Plugin Service returned a malformed JSON response.")
    return payload


async def _mint_identity_edge_biscuit(
    access_token: str,
    account_id: str | None,
    env: str,
) -> str:
    try:
        biscuit_module = importlib.import_module("oai_http_clients.auth.biscuit")
        credential_cls = biscuit_module.SkippedIdentityEdgeBiscuitCredential
    except (ImportError, AttributeError):
        return _mint_identity_edge_biscuit_over_http(access_token, account_id, env)

    credential = credential_cls(access_token, workspace=account_id, env=env)
    try:
        return await credential.get_biscuit()
    finally:
        close = getattr(credential, "close", None)
        if close is not None:
            await _maybe_await(close())


def _mint_identity_edge_biscuit_over_http(
    access_token: str,
    account_id: str | None,
    env: str,
) -> str:
    endpoint = (
        "/authenticate_api" if access_token.startswith(("sk-", "sess-")) else "/authenticate_app_v2"
    )
    headers = {"Authorization": f"Bearer {access_token}"}
    if account_id is not None:
        headers["ChatGPT-Account-ID"] = account_id

    request = Request(
        f"{_identity_edge_base_url(env)}{endpoint}",
        headers=headers,
        method="GET",
    )
    with urlopen(request, timeout=60) as response:
        biscuit = response.headers.get(IDENTITY_EDGE_BISCUIT_HEADER)
        error = response.headers.get("x-openai-authorization-error")
    if error or not biscuit:
        raise UserError(
            f"Identity Edge did not return an SIWC biscuit: {error or 'missing header'}"
        )
    return biscuit


def _identity_edge_base_url(env: str) -> str:
    override = os.getenv(OPENAI_SIWC_IDENTITY_EDGE_URL_ENV)
    if override and override.strip():
        return override.strip().rstrip("/")
    if env.startswith("http://") or env.startswith("https://"):
        return env.rstrip("/")
    if env == "local":
        return f"http://localhost:{os.getenv('IDENTITY_EDGE_PORT', '3141')}"
    if env == "staging":
        return "https://identity-edge-namespaced-default-internal.gateway.unified-0s.internal.api.openai.org"
    if env == "prod":
        return "https://identity-edge-namespaced-default-internal.gateway.unified-4.internal.api.openai.org"
    raise UserError(
        f"Unknown SIWC Identity Edge environment {env!r}. "
        f"Set {OPENAI_SIWC_IDENTITY_EDGE_URL_ENV} to an explicit URL."
    )


def _issuer_without_trailing_slash(issuer: str) -> str:
    return issuer.rstrip("/")


def _create_pkce() -> dict[str, str]:
    code_verifier = secrets.token_urlsafe(32)
    code_challenge = (
        base64.urlsafe_b64encode(hashlib.sha256(code_verifier.encode()).digest())
        .rstrip(b"=")
        .decode()
    )
    return {"code_verifier": code_verifier, "code_challenge": code_challenge}


def _build_chatgpt_siwc_authorize_url(
    *,
    issuer: str,
    client_id: str,
    redirect_uri: str,
    scopes: Sequence[str],
    code_challenge: str,
    state: str,
    extra_params: Mapping[str, str] | None,
) -> str:
    params = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "scope": " ".join(scopes),
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
        "state": state,
        "codex_cli_simplified_flow": "true",
    }
    if extra_params is not None:
        params.update(extra_params)
    return f"{issuer}/oauth/authorize?{urlencode(params)}"


def _oauth_authorization_code_token_exchange(
    *,
    issuer: str,
    client_id: str,
    redirect_uri: str,
    code: str,
    code_verifier: str,
) -> Mapping[str, Any]:
    body = urlencode(
        {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
            "client_id": client_id,
            "code_verifier": code_verifier,
        }
    ).encode("utf-8")
    request = Request(
        f"{issuer}/oauth/token",
        data=body,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    with urlopen(request, timeout=60) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, Mapping):
        raise UserError("SIWC token exchange returned a malformed response.")
    return payload


async def _maybe_await(value: object | Awaitable[object]) -> object:
    if inspect.isawaitable(value):
        return await value
    return value


class _OAuthCallbackServer:
    def __init__(
        self,
        *,
        host: str,
        ports: Sequence[int],
        path: str,
        state: str,
    ) -> None:
        self._host = host
        self._ports = ports
        self._path = path
        self._state = state
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self.redirect_uri: str = ""
        self.authorization_code: asyncio.Future[str]

    def start(self) -> None:
        loop = asyncio.get_running_loop()
        self.authorization_code = loop.create_future()
        callback = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                parsed = urlparse(self.path)
                if parsed.path != callback._path:
                    self._send_text(HTTPStatus.NOT_FOUND, "Not Found")
                    return

                query = parse_qs(parsed.query)
                if query.get("state") != [callback._state]:
                    self._send_text(HTTPStatus.BAD_REQUEST, "State mismatch")
                    return

                if "error" in query:
                    error = query["error"][0]
                    description = query.get("error_description", [""])[0]
                    message = f"{error}: {description}" if description else error
                    loop.call_soon_threadsafe(callback._set_exception, UserError(message))
                    self._send_text(HTTPStatus.BAD_REQUEST, "Sign in with ChatGPT failed")
                    return

                code = query.get("code", [""])[0]
                if not code:
                    self._send_text(HTTPStatus.BAD_REQUEST, "Missing authorization code")
                    return

                loop.call_soon_threadsafe(callback._set_result, code)
                self._send_html(HTTPStatus.OK, callback._success_page())

            def log_message(self, format: str, *args: Any) -> None:
                return None

            def _send_text(self, status: HTTPStatus, body: str) -> None:
                self._send(status, body, "text/plain; charset=utf-8")

            def _send_html(self, status: HTTPStatus, body: str) -> None:
                self._send(status, body, "text/html; charset=utf-8")

            def _send(self, status: HTTPStatus, body: str, content_type: str) -> None:
                encoded = body.encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

        last_error: OSError | None = None
        for port in self._ports:
            try:
                self._server = ThreadingHTTPServer((self._host, port), Handler)
                actual_port = self._server.server_address[1]
                self.redirect_uri = f"http://{self._host}:{actual_port}{self._path}"
                break
            except OSError as exc:
                last_error = exc
        if self._server is None:
            ports = ", ".join(str(port) for port in self._ports)
            raise UserError(f"SIWC callback ports are unavailable: {ports}") from last_error

        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    async def close(self) -> None:
        if self._server is not None:
            await asyncio.to_thread(self._server.shutdown)
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=1)
            self._thread = None
        if hasattr(self, "authorization_code") and not self.authorization_code.done():
            self.authorization_code.cancel()

    def _set_result(self, code: str) -> None:
        if not self.authorization_code.done():
            self.authorization_code.set_result(code)

    def _set_exception(self, error: Exception) -> None:
        if not self.authorization_code.done():
            self.authorization_code.set_exception(error)

    def _success_page(self) -> str:
        return """<!doctype html>
<html>
  <head><meta charset="utf-8" /><title>Signed in</title></head>
  <body style="font-family: ui-sans-serif, system-ui; padding: 32px;">
    <h1>Signed in with ChatGPT</h1>
    <p>You can close this tab and return to your agent run.</p>
  </body>
</html>"""


def _agent_run_request(
    agent: Agent[Any],
    input: str | Sequence[Mapping[str, Any]],
    *,
    cwd: str | None,
    environment: Literal["local", "delegated"],
    dynamic_tools: Sequence[ManagedDynamicTool | Mapping[str, Any]] | None,
) -> dict[str, Any]:
    model = _agent_model(agent)
    instructions = _agent_instructions(agent)
    request: dict[str, Any] = {
        "type": "agent.run",
        "id": f"evt_{uuid.uuid4().hex}",
        "agent": {
            "model": model,
        },
        "environment": {"type": environment},
        "input": _normalize_input(input),
    }
    if instructions is not None:
        request["agent"]["instructions"] = instructions
    if cwd is not None:
        request["workspace"] = {"manifest": {"cwd": cwd}}
    if dynamic_tools is not None:
        request["agent"]["dynamicTools"] = [_dynamic_tool_json(tool) for tool in dynamic_tools]
    return request


def _agent_model(agent: Agent[Any]) -> str:
    if agent.model is None:
        return get_default_model()
    if isinstance(agent.model, str):
        return agent.model
    raise UserError("ManagedRunner.run requires agent.model to be a model name string.")


def _agent_instructions(agent: Agent[Any]) -> str | None:
    instructions = agent.instructions
    if instructions is None:
        return None
    if isinstance(instructions, str):
        return instructions
    raise UserError("ManagedRunner.run currently requires static string instructions.")


def _normalize_input(input: str | Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(input, str):
        return [{"type": "text", "text": input}]
    return [dict(item) for item in input]


def _dynamic_tool_json(tool: ManagedDynamicTool | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(tool, ManagedDynamicTool):
        return tool.to_agent_run_json()
    return dict(tool)


def _agent_message_delta(event: Mapping[str, Any]) -> str | None:
    if event.get("type") != "app_server.event":
        return None
    data = event.get("data")
    if not isinstance(data, Mapping):
        return None
    if data.get("method") != "item/agentMessage/delta":
        return None
    params = data.get("params")
    if not isinstance(params, Mapping):
        return None
    delta = params.get("delta")
    return delta if isinstance(delta, str) else None


def _connect_header_kwargs(headers: Mapping[str, str]) -> dict[str, Mapping[str, str]]:
    params = inspect.signature(websockets.connect).parameters
    if "additional_headers" in params:
        return {"additional_headers": headers}
    return {"extra_headers": headers}
