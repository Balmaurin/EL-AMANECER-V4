"""Shim for training-related utilities expected by tests."""

from typing import Any, Dict


def create_training_config(name: str, params: Dict[str, Any]) -> Dict[str, Any]:
    cfg = {"name": name, "params": params or {}, "type": "shim_training"}
    # Provide a default early_stopping_threshold used by tests
    cfg["params"].setdefault("early_stopping_threshold", 0.01)
    cfg["params"].setdefault("early_stopping_patience", 3)
    return cfg


def validate_training_config(cfg: Dict[str, Any]) -> bool:
    return isinstance(cfg, dict) and "params" in cfg


def create_training_engine(config: Dict[str, Any]) -> Dict[str, Any]:
    return {"engine": "shim_engine", "config": config}
