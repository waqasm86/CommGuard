from __future__ import annotations

import json
from pathlib import Path

from commguard.artifacts import configuration_hash, materialize_calibration_package
from commguard.scope import PROTOTYPE_SCOPE_DECLARATION


def test_configuration_hash_is_order_independent() -> None:
    assert configuration_hash({"a": 1, "b": [2, 3]}) == configuration_hash({"b": [2, 3], "a": 1})


def test_calibration_package_has_required_files_and_scope(tmp_path: Path, monkeypatch) -> None:
    def fake_plot(path: Path, observations: object) -> None:
        assert observations
        path.write_bytes(b"real-data-derived-plot-stub")

    monkeypatch.setattr("commguard.artifacts.prototype._plot_calibration", fake_plot)
    package = materialize_calibration_package(
        tmp_path,
        {
            "status": "supported",
            "result_state": "supported",
            "modern_capture_gate_passed": True,
            "target_sampling_interval_s": 0.5,
            "supported_payload_range_mib": [16, 64, 128],
            "falsification_reasons": [],
            "observations": [
                {
                    "run_id": "run-1",
                    "observation_type": "collective",
                    "payload_mib": 16,
                    "pcie_total_mean_bytes_per_s": 10_000_000,
                    "exit_status": "completed",
                }
            ],
        },
        [{"target_interval_s": 0.5, "median_actual_interval_s": 0.51}],
        environment={"python_version": "3.11"},
        provenance={
            "source_commit": "a" * 40,
            "source_dirty": False,
            "experiment_session_id": "session-1",
        },
        notebook_filename="commguard_calibration_v4.ipynb",
        notebook_sha256="b" * 64,
        configuration={"sampling_interval_s": 0.5},
        development_smoke_only=False,
    )
    required = {
        "calibration_manifest.json",
        "calibration_runs.jsonl",
        "calibration_summary.json",
        "sampling_interval_comparison.csv",
        "sampling_interval_comparison.json",
        "calibration_acceptance.json",
        "calibration_plot.png",
        "environment.json",
        "provenance.json",
        "sha256sums.txt",
        "run_status.json",
    }
    assert {path.name for path in package.iterdir()} == required
    manifest = json.loads((package / "calibration_manifest.json").read_text(encoding="utf-8"))
    assert manifest["prototype_scope"] == "single_node_dual_gpu"
    assert manifest["scope_declaration"] == PROTOTYPE_SCOPE_DECLARATION
    checksums = (package / "sha256sums.txt").read_text(encoding="utf-8")
    assert "calibration_manifest.json" in checksums
    assert "sha256sums.txt" not in checksums
