"""Lightweight adapters utilities used by tests (shims).

These functions provide deterministic, test-friendly behavior and avoid heavy
dependencies. They are intentionally minimal to satisfy unit/integration tests.
"""

from typing import Any, Dict


def create_adapter_config(name: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Return a normalized adapter config dict."""
    return {"name": name, "params": params or {}, "version": "test-shim-1.0"}


def validate_adapter_config(cfg: Dict[str, Any]) -> bool:
    """Basic validation: must have name and params."""
    return isinstance(cfg, dict) and "name" in cfg and "params" in cfg


def create_adapter_state(name: str) -> Dict[str, Any]:
    return {"adapter": name, "state": "initialized"}
