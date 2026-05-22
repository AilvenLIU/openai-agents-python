#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import signal
import socket
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import websockets

from agents import (
    Agent,
    ChatGPTSIWCInteractiveProvider,
    CodexHostSIWCProvider,
    LocalSIWCInteractiveProvider,
    ManagedDynamicTool,
    ManagedRunner,
    SIWCAuth,
    SIWCIdentity,
)

IDENTITY_EDGE_BISCUIT_HEADER = "x-openai-authorization"
CHATGPT_ACCOUNT_ID_HEADER = "chatgpt-account-id"
THREAD_START_ID = "managedagentsapi:thread-start"
TURN_START_ID = "managedagentsapi:turn-start"
INITIALIZE_ID = "managedagentsapi:initialize"
BYOK_MODEL_PROVIDER = "managedagentsapi-byok"


@dataclass
class AuthDemoCase:
    name: str
    auth: SIWCAuth
    interactive_provider: LocalSIWCInteractiveProvider | None = None
    auto_complete_ui: bool = False
    print_ui_url: bool = False


@dataclass
class UpstreamExchange:
    request_headers: dict[str, str]
    frames: list[dict[str, Any]] = field(default_factory=list)

    @property
    def thread_start(self) -> dict[str, Any] | None:
        for frame in self.frames:
            if frame.get("id") == THREAD_START_ID:
                return frame
        return None


class FakeCloudRouter:
    def __init__(self) -> None:
        self.exchanges: list[UpstreamExchange] = []

    async def handler(self, websocket: Any) -> None:
        exchange = UpstreamExchange(request_headers=_websocket_headers(websocket))
        self.exchanges.append(exchange)
        thread_id = f"thread_demo_{len(self.exchanges)}"
        turn_id = f"turn_demo_{len(self.exchanges)}"

        try:
            async for message in websocket:
                frame = json.loads(message)
                exchange.frames.append(frame)
                frame_id = frame.get("id")
                method = frame.get("method")

                if frame_id == INITIALIZE_ID and method == "initialize":
                    await websocket.send(json.dumps({"id": frame_id, "result": {}}))
                elif method == "initialized":
                    pass
                elif frame_id == THREAD_START_ID and method == "thread/start":
                    await websocket.send(
                        json.dumps({"id": frame_id, "result": {"thread": {"id": thread_id}}})
                    )
                elif frame_id == TURN_START_ID and method == "turn/start":
                    await websocket.send(
                        json.dumps({"id": frame_id, "result": {"turn": {"id": turn_id}}})
                    )
                    await websocket.send(
                        json.dumps(
                            {
                                "method": "item/agentMessage/delta",
                                "params": {"delta": f"{thread_id}: hello from fake cloud-router"},
                            }
                        )
                    )
                    await websocket.send(
                        json.dumps(
                            {
                                "method": "turn/completed",
                                "params": {"turn": {"id": turn_id}},
                            }
                        )
                    )
        except websockets.exceptions.ConnectionClosed:
            return


def auth_cases(args: argparse.Namespace) -> list[AuthDemoCase]:
    if args.real_siwc:
        host_provider = CodexHostSIWCProvider(
            auth_path=args.codex_auth_json,
            identity_edge_env=args.identity_edge_env,
        )
        interactive_provider = ChatGPTSIWCInteractiveProvider(
            issuer=args.siwc_issuer,
            client_id=args.siwc_client_id,
            identity_edge_env=args.identity_edge_env,
        )
        cases = [
            AuthDemoCase(
                name="from_host_real",
                auth=SIWCAuth.from_host(provider=host_provider),
            ),
            AuthDemoCase(
                name="auto_real",
                auth=SIWCAuth.auto(
                    host_provider=host_provider,
                    interactive_provider=interactive_provider,
                    interactive=True,
                ),
            ),
            AuthDemoCase(
                name="interactive_real",
                auth=SIWCAuth.interactive(provider=interactive_provider),
            ),
        ]
        return _filter_cases(cases, args.case)

    interactive_auth = _interactive_auth_case(args)
    cases = [
        AuthDemoCase(
            name="from_host",
            auth=SIWCAuth.from_host(
                provider=lambda: SIWCIdentity(
                    identity_edge_biscuit="biscuit:from_host",
                    chatgpt_account_id="acct_from_host",
                )
            ),
        ),
        AuthDemoCase(
            name="auto_interactive_fallback",
            auth=SIWCAuth.auto(
                host_provider=lambda: None,
                interactive_provider=lambda: SIWCIdentity(
                    identity_edge_biscuit="biscuit:auto_interactive",
                    chatgpt_account_id="acct_auto_interactive",
                ),
                interactive=True,
            ),
        ),
        interactive_auth,
    ]
    return _filter_cases(cases, args.case)


def _filter_cases(cases: list[AuthDemoCase], selected: list[str] | None) -> list[AuthDemoCase]:
    if not selected:
        return cases
    selected_set = set(selected)
    filtered = [case for case in cases if case.name in selected_set]
    if not filtered:
        available = ", ".join(case.name for case in cases)
        raise ValueError(f"no matching --case values; available cases: {available}")
    return filtered


def _interactive_auth_case(args: argparse.Namespace) -> AuthDemoCase:
    if args.interactive_ui:
        provider = LocalSIWCInteractiveProvider(open_browser=not args.no_open_browser)
        return AuthDemoCase(
            name="interactive_ui",
            auth=SIWCAuth.interactive(provider=provider),
            interactive_provider=provider,
            auto_complete_ui=args.auto_complete_ui,
            print_ui_url=args.no_open_browser and not args.auto_complete_ui,
        )

    return AuthDemoCase(
        name="interactive",
        auth=SIWCAuth.interactive(
            provider=lambda: SIWCIdentity(
                identity_edge_biscuit="biscuit:interactive",
                chatgpt_account_id="acct_interactive",
            )
        ),
    )


async def run_case(endpoint: str, case: AuthDemoCase):
    agent = Agent(
        name="Managed SIWC demo",
        model="gpt-5.5",
        instructions="Reply through managedagentsapi for the SIWC SDK demo.",
    )
    return await ManagedRunner.run(
        agent,
        f"Run the {case.name} case.",
        auth=case.auth,
        endpoint=endpoint,
        cwd="/tmp",
        dynamic_tools=[
            ManagedDynamicTool(
                namespace="demo",
                name="lookup_profile",
                description="Placeholder plugin-style dynamic tool declaration.",
                input_schema={
                    "type": "object",
                    "properties": {"account_id": {"type": "string"}},
                },
                defer_loading=True,
            )
        ],
    )


async def run_demo(args: argparse.Namespace) -> int:
    managed_port = args.managed_port or _free_port()
    cloud_router_port = args.cloud_router_port or _free_port()
    cloud_router = FakeCloudRouter()
    output: list[str] = []

    async with websockets.serve(cloud_router.handler, "127.0.0.1", cloud_router_port):
        process = await _start_managedagentsapi(
            Path(args.managedagentsapi_bin).resolve(),
            managed_port,
            cloud_router_port,
        )
        stdout_task = asyncio.create_task(_capture_output(process.stdout, "stdout", output))
        stderr_task = asyncio.create_task(_capture_output(process.stderr, "stderr", output))

        try:
            await _wait_for_tcp("127.0.0.1", managed_port, timeout=args.startup_timeout)
            endpoint = f"ws://127.0.0.1:{managed_port}/v1/agents/runs"
            print(f"managedagentsapi endpoint: {endpoint}")
            print(f"fake cloud-router endpoint: ws://127.0.0.1:{cloud_router_port}/")

            for case in auth_cases(args):
                before = len(cloud_router.exchanges)
                result = await run_case_with_optional_ui_completion(endpoint, case)
                if len(cloud_router.exchanges) != before + 1:
                    raise RuntimeError(f"expected one upstream exchange for {case.name}")
                exchange = cloud_router.exchanges[-1]
                print_case(case, result.final_output, exchange)
            return 0
        finally:
            await _stop_process(process)
            await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
            if args.show_service_logs:
                print("\nmanagedagentsapi logs:")
                for line in output:
                    print(f"  {line}")


async def run_case_with_optional_ui_completion(endpoint: str, case: AuthDemoCase):
    if case.interactive_provider is None:
        return await run_case(endpoint, case)

    task = asyncio.create_task(run_case(endpoint, case))
    try:
        authorize_url = await _wait_for_authorize_url(case.interactive_provider)
        if case.print_ui_url:
            print(f"\nOpen this local sign-in URL to continue: {authorize_url}")
        complete_url = await _wait_for_complete_url(case.interactive_provider)
        if case.auto_complete_ui:
            await asyncio.to_thread(lambda: urlopen(complete_url, timeout=5).read())
        return await task
    finally:
        if not task.done():
            task.cancel()


def print_case(case: AuthDemoCase, final_output: str, exchange: UpstreamExchange) -> None:
    print(f"\n=== {case.name} ===")
    print("ManagedRunner final_output:")
    print(f"  {final_output}")
    print("managedagentsapi -> cloud-router identity headers:")
    for name in [IDENTITY_EDGE_BISCUIT_HEADER, CHATGPT_ACCOUNT_ID_HEADER]:
        value = exchange.request_headers.get(name)
        print(f"  {name}: {_redact(value)}")

    thread_start = exchange.thread_start
    if thread_start is None:
        print("thread/start: <missing>")
        return
    params = thread_start["params"]
    lane = (
        "SIWC identity-edge lane"
        if params.get("modelProvider") is None and params.get("config") is None
        else f"unexpected lane: {params.get('modelProvider')!r}"
    )
    print(f"thread/start auth lane: {lane}")
    print(f"thread/start dynamicTools count: {len(params.get('dynamicTools') or [])}")


def _websocket_headers(websocket: Any) -> dict[str, str]:
    request = getattr(websocket, "request", None)
    if request is not None:
        headers = getattr(request, "headers", None)
        if headers is not None:
            return {name.lower(): value for name, value in headers.raw_items()}

    headers = getattr(websocket, "request_headers", None)
    if headers is not None:
        return {name.lower(): value for name, value in headers.raw_items()}
    return {}


def _redact(value: str | None) -> str:
    if value is None:
        return "<missing>"
    return f"<redacted len={len(value)}>"


async def _wait_for_complete_url(provider: LocalSIWCInteractiveProvider) -> str:
    for _ in range(500):
        if provider.complete_url is not None:
            return provider.complete_url
        await asyncio.sleep(0.01)
    raise TimeoutError("interactive UI did not publish callback URL")


async def _wait_for_authorize_url(provider: LocalSIWCInteractiveProvider) -> str:
    for _ in range(500):
        if provider.authorize_url is not None:
            return provider.authorize_url
        await asyncio.sleep(0.01)
    raise TimeoutError("interactive UI did not publish sign-in URL")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


async def _wait_for_tcp(host: str, port: int, timeout: float) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while True:
        try:
            reader, writer = await asyncio.open_connection(host, port)
            writer.close()
            await writer.wait_closed()
            return
        except OSError:
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError(f"timed out waiting for {host}:{port}") from None
            await asyncio.sleep(0.05)


async def _start_managedagentsapi(
    binary: Path,
    managed_port: int,
    cloud_router_port: int,
) -> asyncio.subprocess.Process:
    env = os.environ.copy()
    env.update(
        {
            "MANAGEDAGENTSAPI_LISTEN_PORT": str(managed_port),
            "MANAGEDAGENTSAPI_CLOUD_ROUTER_WS_URL": f"ws://127.0.0.1:{cloud_router_port}/",
            "MANAGEDAGENTSAPI_CLOSE_ON_IDLE": "true",
        }
    )
    return await asyncio.create_subprocess_exec(
        str(binary),
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )


async def _capture_output(
    stream: asyncio.StreamReader | None,
    label: str,
    output: list[str],
) -> None:
    if stream is None:
        return
    while True:
        line = await stream.readline()
        if not line:
            return
        output.append(f"{label}: {line.decode(errors='replace').rstrip()}")


async def _stop_process(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
    try:
        await asyncio.wait_for(process.wait(), timeout=3)
    except TimeoutError:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        await process.wait()


def parse_args(argv: list[str]) -> argparse.Namespace:
    default_binary = (
        Path.home()
        / "code"
        / "openai"
        / "api"
        / "managedagentsapi"
        / "target"
        / "debug"
        / "managedagentsapi"
    )
    parser = argparse.ArgumentParser(
        description="Run Agents SDK SIWC auth modes through real managedagentsapi."
    )
    parser.add_argument("--managedagentsapi-bin", default=str(default_binary))
    parser.add_argument("--managed-port", type=int, default=None)
    parser.add_argument("--cloud-router-port", type=int, default=None)
    parser.add_argument("--startup-timeout", type=float, default=30)
    parser.add_argument("--show-service-logs", action="store_true")
    parser.add_argument(
        "--case",
        action="append",
        default=None,
        help="Run only a named auth case. Repeat to select multiple cases.",
    )
    parser.add_argument(
        "--interactive-ui",
        action="store_true",
        help="Use the loopback Sign in with ChatGPT demo page for the interactive case.",
    )
    parser.add_argument(
        "--no-open-browser",
        action="store_true",
        help="Do not open a browser for --interactive-ui. Useful with --auto-complete-ui.",
    )
    parser.add_argument(
        "--auto-complete-ui",
        action="store_true",
        help="Automatically hit the loopback callback for --interactive-ui verification.",
    )
    parser.add_argument(
        "--real-siwc",
        action="store_true",
        help=(
            "Use real ChatGPT/Codex SIWC providers instead of demo identities. "
            "This reads Codex host auth for from_host/auto and opens browser OAuth "
            "for the interactive case."
        ),
    )
    parser.add_argument(
        "--codex-auth-json",
        default=None,
        help="Optional Codex auth.json path for --real-siwc host identity.",
    )
    parser.add_argument(
        "--identity-edge-env",
        default=None,
        help="Identity Edge environment for real SIWC biscuit exchange, for example prod or staging.",
    )
    parser.add_argument(
        "--siwc-issuer",
        default=None,
        help="Optional AuthAPI issuer override for real interactive SIWC.",
    )
    parser.add_argument(
        "--siwc-client-id",
        default=None,
        help="Optional OAuth client id override for real interactive SIWC.",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if not Path(args.managedagentsapi_bin).exists():
        print(
            f"managedagentsapi binary not found: {args.managedagentsapi_bin}\n"
            "Build it first with: cargo build --manifest-path "
            "/Users/dannyzhang/code/openai/api/managedagentsapi/Cargo.toml --bin "
            "managedagentsapi",
            file=sys.stderr,
        )
        return 1
    return asyncio.run(run_demo(args))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
