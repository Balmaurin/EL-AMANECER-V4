"""Shim for training router utilities used by tests."""

from typing import Any, Dict


def route_for_training(job: Dict[str, Any]) -> Dict[str, Any]:
    return {"route": "default", "job": job}


def calculate_route_priority(job: Dict[str, Any]) -> int:
    return job.get("priority", 0)
