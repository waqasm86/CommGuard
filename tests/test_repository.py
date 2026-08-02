from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

from commguard.orchestrator import estimate_matrix
from commguard.reporting import generate_report
from commguard.workloads import list_workloads

ROOT = Path(__file__).resolve().parents[1]
CANONICAL_NOTEBOOKS = [
    "commguard_calibration_v3.ipynb",
    "commguard_benign_corpus_v2.ipynb",
    "commguard_detector_evaluation_v2.ipynb",
    "commguard_adversarial_redteam_v1.ipynb",
]


def test_kaggle_snapshot_is_verbatim() -> None:
    digest = hashlib.sha256((ROOT / "pip-kaggle-list.txt").read_bytes()).hexdigest()
    assert digest == "f45bb00d807da387d944f692b7626f91737056c735e4681127415ecdccfa0dce"
    assert len((ROOT / "pip-kaggle-list.txt").read_text().splitlines()) == 935


def test_canonical_notebook_inventory_is_valid_and_unexecuted() -> None:
    notebook_root = ROOT / "notebooks"
    inventory = json.loads((notebook_root / "canonical_notebooks.json").read_text(encoding="utf-8"))
    names = inventory["canonical_notebooks"]
    assert inventory["schema_version"] == 1
    assert names == CANONICAL_NOTEBOOKS
    assert len(names) == len(set(names))
    assert all("-" not in name and name.endswith(".ipynb") for name in names)
    assert (ROOT / "docs/notebook-policy.md").is_file()

    for name in names:
        notebook = json.loads((notebook_root / name).read_text(encoding="utf-8"))
        assert notebook["nbformat"] == 4
        assert notebook["cells"]
        code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        assert all(cell.get("execution_count") is None for cell in code_cells)
        assert all(not cell.get("outputs") for cell in code_cells)
        source = "\n".join(
            cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])
            for cell in code_cells
        )
        assert "class TelemetryCollector" not in source
        assert "class ArtifactStore" not in source
        assert "drive.google.com/file/d/" not in source
        assert "REVIEWED_COMMIT" in source
        assert "--no-build-isolation" in source
        assert "--no-deps" in source
        assert 'checkout", "main' not in source
        assert "@main" not in source
        assert "NEXT STEP:" in source
        assert "SHA-256" in source
        assert notebook["metadata"]["commguard"]["required_accelerator"] == "two NVIDIA T4 GPUs"
        for cell in code_cells:
            compile("".join(cell["source"]), f"{name}:{cell['id']}", "exec")
        all_source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
        assert "## Results" in all_source
        assert "not executed" in all_source
        assert "def run_experiment" not in all_source
        assert "def extract_features" not in all_source

    downstream = names[1:]
    for name in downstream:
        source = (notebook_root / name).read_text(encoding="utf-8")
        assert "restore_archive" in source
        assert "EXPECTED_INPUT_SHA256" in source
    adversarial = (notebook_root / names[-1]).read_text(encoding="utf-8")
    assert "RUN_ADVERSARIAL_PILOT = False" in adversarial
    assert "ADVERSARIAL_HUMAN_APPROVAL = False" in adversarial


def test_canonical_notebooks_match_their_generator() -> None:
    result = subprocess.run(
        [sys.executable, "tools/generate_canonical_notebooks.py", "--check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


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
