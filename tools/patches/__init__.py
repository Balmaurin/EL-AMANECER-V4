"""
Hotpatch System Module
======================

Sistema de hotpatching para auto-repair en vivo sin downtime.
"""

from .hotpatch_system import (
    HotpatchSystem,
    Patch,
    PatchValidation,
    create_parameter_interpolation_patch,
    create_layer_swap_patch,
    create_neuron_patching_patch,
    validate_patch_functionality,
)

__all__ = [
    "HotpatchSystem",
    "Patch",
    "PatchValidation",
    "create_parameter_interpolation_patch",
    "create_layer_swap_patch",
    "create_neuron_patching_patch",
    "validate_patch_functionality",
]
