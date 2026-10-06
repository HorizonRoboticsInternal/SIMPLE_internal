"""Compatibility exports for the shared policy observation builder."""

from holomotion_policy_core.obs_builder import (
    PolicyObsBuilder,
    get_gravity_orientation,
)

__all__ = ["PolicyObsBuilder", "get_gravity_orientation"]
