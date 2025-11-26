"""LLM engine shims for tests.

This package provides lightweight implementations of the submodules expected
by the test-suite. They are intentionally small and deterministic.
"""

__all__ = [
    "training",
    "data_preparation",
    "gguf_integration",
    "lora_training",
    "training_router",
    "training_orchestrator",
    "training_deps",
]
