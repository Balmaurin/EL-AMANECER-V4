"""Shim for data preparation utilities expected by tests."""

from typing import Any, Dict


def create_data_preparation_config(
    source: str, options: Dict[str, Any]
) -> Dict[str, Any]:
    return {"source": source, "options": options or {}, "prepared": True}


def validate_data_prep_config(cfg: Dict[str, Any]) -> bool:
    return isinstance(cfg, dict) and "source" in cfg
