#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import asdict
from typing import Any

from agents import (
    ChatGPTSIWCInteractiveProvider,
    CodexPlugin,
    CodexPluginCacheClient,
    CodexPluginServiceClient,
    SIWCAuth,
    UserError,
    format_codex_plugins,
)


async def run(args: argparse.Namespace) -> int:
    plugins = await _list_plugins(args)

    if args.json:
        print(json.dumps([_plugin_to_json(plugin) for plugin in plugins], indent=2))
    else:
        print(format_codex_plugins(plugins))
    return 0


async def _list_plugins(args: argparse.Namespace) -> list[CodexPlugin]:
    source = args.source
    if source == "local" or (source == "auto" and _plugin_service_url(args) is None):
        return _list_local_plugins(args)

    try:
        return await _list_service_plugins(args)
    except UserError as exc:
        if source != "auto" or not _should_fallback_to_local(exc):
            raise
        print(
            f"warning: {exc}; falling back to the local Codex plugin cache.",
            file=sys.stderr,
        )
        return _list_local_plugins(args)


async def _list_service_plugins(args: argparse.Namespace) -> list[CodexPlugin]:
    if args.interactive_siwc:
        identity = await SIWCAuth.interactive(
            provider=ChatGPTSIWCInteractiveProvider(
                issuer=args.siwc_issuer,
                client_id=args.siwc_client_id,
                identity_edge_env=args.identity_edge_env,
            )
        ).resolve()
        client = CodexPluginServiceClient.from_siwc_identity(
            identity,
            plugin_service_url=args.plugin_service_url,
        )
    else:
        client = CodexPluginServiceClient(
            plugin_service_url=args.plugin_service_url,
            auth_path=args.codex_auth_json,
        )

    return await client.list_plugins(
        include_global_installed=not args.no_global_installed,
        include_workspace_installed=not args.no_workspace_installed,
        include_shared_workspace=not args.no_shared_workspace,
        include_created_workspace=not args.no_created_workspace,
        include_available=args.include_available,
        limit=args.limit,
    )


def _list_local_plugins(args: argparse.Namespace) -> list[CodexPlugin]:
    return CodexPluginCacheClient(
        config_path=args.codex_config,
        plugin_cache_dir=args.plugin_cache_dir,
    ).list_plugins(include_disabled=args.include_disabled)


def _plugin_service_url(args: argparse.Namespace) -> str | None:
    return args.plugin_service_url or os.getenv("OPENAI_CODEX_PLUGIN_SERVICE_URL")


def _should_fallback_to_local(error: UserError) -> bool:
    message = str(error)
    return (
        "Plugin Service request failed" in message
        or "requires a Plugin Service URL" in message
        or "Connection refused" in message
    )


def _plugin_to_json(plugin: CodexPlugin) -> dict[str, Any]:
    payload = asdict(plugin)
    payload["raw"] = dict(plugin.raw)
    return payload


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="List Codex plugins available to the signed-in ChatGPT identity."
    )
    parser.add_argument(
        "--source",
        choices=("auto", "service", "local"),
        default="auto",
        help=(
            "Inventory source. auto uses Plugin Service when reachable and otherwise local "
            "Codex config/cache; service is strict; local never calls Plugin Service."
        ),
    )
    parser.add_argument(
        "--plugin-service-url",
        default=None,
        help=(
            "Plugin Service origin, for example http://127.0.0.1:3001 when port-forwarding "
            "plugin-service. Defaults to OPENAI_CODEX_PLUGIN_SERVICE_URL."
        ),
    )
    parser.add_argument(
        "--codex-auth-json",
        default=None,
        help="Optional Codex auth.json path. Defaults to CODEX_AUTH_PATH, CODEX_HOME/auth.json, or ~/.codex/auth.json.",
    )
    parser.add_argument(
        "--codex-config",
        default=None,
        help="Optional Codex config.toml path for --source local/auto fallback.",
    )
    parser.add_argument(
        "--plugin-cache-dir",
        default=None,
        help="Optional Codex plugin cache directory for --source local/auto fallback.",
    )
    parser.add_argument(
        "--include-disabled",
        action="store_true",
        help="Include disabled entries from the local Codex plugin config.",
    )
    parser.add_argument(
        "--interactive-siwc",
        action="store_true",
        help="Open the real Sign in with ChatGPT browser flow, then list plugins from that identity.",
    )
    parser.add_argument(
        "--identity-edge-env",
        default=None,
        help="Identity Edge environment for --interactive-siwc, for example prod or staging.",
    )
    parser.add_argument(
        "--siwc-issuer",
        default=None,
        help="Optional AuthAPI issuer override for --interactive-siwc.",
    )
    parser.add_argument(
        "--siwc-client-id",
        default=None,
        help="Optional OAuth client id override for --interactive-siwc.",
    )
    parser.add_argument(
        "--limit", type=int, default=100, help="Page size for Plugin Service calls."
    )
    parser.add_argument(
        "--include-available",
        action="store_true",
        help="Also include directory plugins that are available but not necessarily installed.",
    )
    parser.add_argument("--no-global-installed", action="store_true")
    parser.add_argument("--no-workspace-installed", action="store_true")
    parser.add_argument("--no-shared-workspace", action="store_true")
    parser.add_argument("--no-created-workspace", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print normalized plugin JSON.")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        return asyncio.run(run(args))
    except UserError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
