"""Shim for training dependency resolution used by tests."""

from typing import Any, Dict, List


def resolve_dependencies(package_list: List[str]) -> Dict[str, Any]:
    return {"resolved": package_list, "status": "ok"}


def validate_package_security(pkg: str) -> bool:
    # Very lightweight check
    return ".." not in pkg and "http" not in pkg
