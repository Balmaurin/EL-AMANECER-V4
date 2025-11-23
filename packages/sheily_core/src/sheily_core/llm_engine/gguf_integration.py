"""Shim for GGUF related utilities used by tests."""

from typing import Any, Dict


def create_gguf_model_config(
    base_model: str, options: Dict[str, Any]
) -> Dict[str, Any]:
    return {"base_model": base_model, "options": options or {}, "gguf": True}


def validate_gguf_safety(cfg: Dict[str, Any]) -> bool:
    return isinstance(cfg, dict) and cfg.get("gguf", False) is True
