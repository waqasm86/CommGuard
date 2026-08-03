"""Authoritative scope metadata for the Kaggle research prototype."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

PROTOTYPE_SCOPE_DECLARATION = (
    "CommGuard’s Kaggle workflow is a single-node, dual-NVIDIA-T4 research "
    "prototype. It validates experimental methodology and software behavior on "
    "two local GPU ranks. It does not establish generalization to two physical "
    "8-GPU nodes, NVLink/NVSwitch fabrics, RoCE or InfiniBand networks, large "
    "frontier-model workloads, or production treaty-verification deployments."
)

PROTOTYPE_SCOPE_METADATA: dict[str, Any] = {
    "prototype_scope": "single_node_dual_gpu",
    "physical_node_count": 1,
    "gpu_count": 2,
    "target_gpu_model": "Tesla T4",
    "multi_node_validated": False,
    "server_grade_interconnect_validated": False,
    "production_ready": False,
    "scope_declaration": PROTOTYPE_SCOPE_DECLARATION,
}


def with_prototype_scope(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy of *payload* carrying the authoritative scope fields."""
    result = dict(payload)
    for key, value in PROTOTYPE_SCOPE_METADATA.items():
        existing = result.get(key, value)
        if existing != value:
            raise ValueError(f"scope field {key!r} conflicts with the CommGuard prototype scope")
        result[key] = value
    return result
