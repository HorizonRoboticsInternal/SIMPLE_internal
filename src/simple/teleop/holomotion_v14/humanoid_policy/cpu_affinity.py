"""Compatibility exports for the shared CPU-affinity helpers."""

from holomotion_policy_core.cpu_affinity import (
    parse_cpu_affinity,
    set_thread_cpu_affinity,
)

__all__ = ["parse_cpu_affinity", "set_thread_cpu_affinity"]
