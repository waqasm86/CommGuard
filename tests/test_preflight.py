from __future__ import annotations

import pytest

from commguard import preflight
from commguard.exceptions import ReadinessError
from commguard.provenance import ProvenanceContext
from commguard.schemas import FIELD_UNITS, FieldReading, validate_artifact


def gpus():
    return [
        {
            "index": index,
            "name": "NVIDIA T4",
            "uuid": f"GPU-{index}",
            "pci_bus_id": f"0000:0{index}:00.0",
            "memory_total_mib": 15109,
            "driver_version": "570",
            "compute_capability": "7.5",
        }
        for index in (0, 1)
    ]


def torch_details():
    return {
        "available": True,
        "version": "2.10.0+cu128",
        "cuda_available": True,
        "cuda_runtime": "12.8",
        "device_count": 2,
        "distributed_available": True,
        "nccl_available": True,
        "nccl_version": "2.27.5",
        "peer_access": {"0->1": True, "1->0": True},
    }


def test_strict_preflight_accepts_two_t4s(monkeypatch) -> None:
    monkeypatch.setattr(preflight, "_gpu_inventory", lambda: (gpus(), {"exit_code": 0}))
    monkeypatch.setattr(preflight, "_torch_details", torch_details)
    monkeypatch.setattr(
        preflight,
        "_command",
        lambda args, timeout=10: {"command": args, "exit_code": 0, "stdout": "", "stderr": ""},
    )
    report = preflight.check_environment(strict=True)
    assert report["strict_ready"]
    assert len(report["session_fingerprint"]) == 64


def test_preflight_persists_true_session_and_environment_ids(monkeypatch) -> None:
    monkeypatch.setattr(preflight, "_gpu_inventory", lambda: (gpus(), {"exit_code": 0}))
    monkeypatch.setattr(preflight, "_torch_details", torch_details)
    monkeypatch.setattr(
        preflight,
        "_command",
        lambda args, timeout=10: {"command": args, "exit_code": 0, "stdout": "", "stderr": ""},
    )
    context = ProvenanceContext(
        experiment_session_id="session-a",
        collection_id="collection-a",
        corpus_id="corpus-a",
        node_id="node-0",
        source_commit="commit-a",
        source_dirty=True,
        notebook_version=None,
        input_archive_sha256=None,
        random_seed=1337,
    )

    report = preflight.check_environment(strict=True, provenance=context)

    assert report["experiment_session_id"] == "session-a"
    assert report["node_id"] == "node-0"
    assert report["environment_fingerprint"] == report["session_fingerprint"]
    assert report["source_dirty"] is True
    validate_artifact(report)


def test_strict_preflight_rejects_single_gpu(monkeypatch) -> None:
    monkeypatch.setattr(preflight, "_gpu_inventory", lambda: (gpus()[:1], {"exit_code": 0}))
    monkeypatch.setattr(preflight, "_torch_details", lambda: {**torch_details(), "device_count": 1})
    monkeypatch.setattr(
        preflight,
        "_command",
        lambda args, timeout=10: {"command": args, "exit_code": 0, "stdout": "", "stderr": ""},
    )
    with pytest.raises(ReadinessError, match="exactly_two_gpus"):
        preflight.check_environment(strict=True)


def test_telemetry_capability_preserves_nullable_value(monkeypatch) -> None:
    class Backend:
        def initialize(self):
            pass

        def shutdown(self):
            pass

        def device_count(self):
            return 1

        def device_uuid(self, index):
            return "GPU-0"

        def read_fields(self, index):
            return {
                "pcie_tx_bytes_per_s": FieldReading(
                    value=None,
                    unit=FIELD_UNITS["pcie_tx_bytes_per_s"],
                    supported=False,
                    error="NVML_ERROR_NOT_SUPPORTED",
                )
            }

    import commguard.telemetry

    monkeypatch.setattr(commguard.telemetry, "NvmlBackend", Backend)
    capabilities = preflight._telemetry_capabilities()
    field = capabilities["devices"][0]["fields"]["pcie_tx_bytes_per_s"]
    assert field == {
        "value": None,
        "supported": False,
        "unit": "bytes/second",
        "error": "NVML_ERROR_NOT_SUPPORTED",
    }
