from __future__ import annotations

import hashlib
import json
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

from commguard.orchestrator import estimate_matrix
from commguard.reporting import generate_report
from commguard.workloads import list_workloads

ROOT = Path(__file__).resolve().parents[1]


def test_kaggle_snapshot_is_verbatim() -> None:
    digest = hashlib.sha256((ROOT / "pip-kaggle-list.txt").read_bytes()).hexdigest()
    assert digest == "f45bb00d807da387d944f692b7626f91737056c735e4681127415ecdccfa0dce"
    assert len((ROOT / "pip-kaggle-list.txt").read_text().splitlines()) == 935


def test_notebook_is_valid_json_and_has_no_sdk_implementation() -> None:
    notebook = json.loads((ROOT / "notebooks/commguard_dual_t4.ipynb").read_text())
    assert notebook["nbformat"] == 4
    source = "\n".join(
        "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"
    )
    assert "pip', 'install', '--no-build-isolation', '--no-deps'" in source
    assert "class TelemetryCollector" not in source


def test_core_install_has_no_forced_dependencies() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert project["project"]["name"] == "commguard"
    assert project["project"]["dependencies"] == []


def test_workload_matrix_includes_required_families() -> None:
    workloads = list_workloads()
    modes = {config["mode"] for config in workloads.values()}
    assert {
        "calibration",
        "ddp_train",
        "inference_independent",
        "control_compute",
        "control_host_transfer",
        "control_model_load",
    } <= modes
    assert estimate_matrix("standard", repetitions=3)["estimated_gpu_minutes"] > 0


def test_empty_report_does_not_fabricate_results(tmp_path) -> None:
    report = generate_report(tmp_path)
    assert "no performance claim is made" in report
    assert "Calibration has not been recorded" in report
    assert "independent, unofficial" in report
