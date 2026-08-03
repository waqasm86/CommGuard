from __future__ import annotations

import pytest

from commguard.environment.preflight import check_environment
from commguard.scope import (
    PROTOTYPE_SCOPE_DECLARATION,
    PROTOTYPE_SCOPE_METADATA,
    with_prototype_scope,
)


def test_authoritative_scope_metadata_is_complete() -> None:
    assert PROTOTYPE_SCOPE_DECLARATION.startswith("CommGuard’s Kaggle workflow")
    assert PROTOTYPE_SCOPE_METADATA == {
        "prototype_scope": "single_node_dual_gpu",
        "physical_node_count": 1,
        "gpu_count": 2,
        "target_gpu_model": "Tesla T4",
        "multi_node_validated": False,
        "server_grade_interconnect_validated": False,
        "production_ready": False,
        "scope_declaration": PROTOTYPE_SCOPE_DECLARATION,
    }


def test_scope_conflicts_are_rejected() -> None:
    with pytest.raises(ValueError, match="conflicts"):
        with_prototype_scope({"physical_node_count": 2})


def test_environment_report_carries_scope(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("commguard.environment.preflight._gpu_inventory", lambda: ([], {}))
    monkeypatch.setattr(
        "commguard.environment.preflight._torch_details",
        lambda: {"cuda_available": False, "device_count": 0, "nccl_available": False},
    )
    monkeypatch.setattr(
        "commguard.environment.preflight._telemetry_capabilities",
        lambda: {"available": False, "devices": [], "error": "test"},
    )
    report = check_environment(output=tmp_path)
    assert report["prototype_scope"] == "single_node_dual_gpu"
    assert report["scope_declaration"] == PROTOTYPE_SCOPE_DECLARATION
