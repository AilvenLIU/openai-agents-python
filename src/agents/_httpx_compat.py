from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Any, cast


def _load_legacy_httpx() -> ModuleType | None:
    try:
        return import_module("httpx")
    except ModuleNotFoundError as exc:
        if exc.name != "httpx":
            raise
        return None


LEGACY_HTTPX = _load_legacy_httpx()


def legacy_httpx_types(*names: str) -> tuple[type[Any], ...]:
    if LEGACY_HTTPX is None:
        return ()
    return tuple(cast(type[Any], getattr(LEGACY_HTTPX, name)) for name in names)


def require_legacy_httpx() -> ModuleType:
    if LEGACY_HTTPX is None:  # pragma: no cover - MCP v1 declares the dependency
        raise ImportError("The installed integration requires the legacy httpx package.")
    return LEGACY_HTTPX
