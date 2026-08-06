from __future__ import annotations

import hashlib
import json
import re
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
    "commguard_calibration_v4.ipynb",
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
    entries = inventory["canonical_notebooks"]
    names = [entry["filename"] for entry in entries]
    assert inventory["schema_version"] == 2
    assert inventory["policy_version"] == "commguard-notebook-policy-v4"
    assert names == CANONICAL_NOTEBOOKS
    assert [entry["order"] for entry in entries] == [1, 2, 3, 4]
    assert all(entry["role"].strip() for entry in entries)
    assert len(names) == len(set(names))
    assert all("-" not in name and name.endswith(".ipynb") for name in names)
    assert (ROOT / "docs/notebook-policy.md").is_file()
    assert inventory["historical_notebooks"] == [
        {
            "filename": "commguard_calibration_v3.ipynb",
            "role": (
                "historical 0.5-second calibration retained for provenance; "
                "not the active downstream gate"
            ),
        }
    ]
    assert inventory["diagnostic_notebooks"] == [
        {
            "filename": "diagnostics/commguard_calibration_v4_sampling_study.ipynb",
            "role": "diagnostic sampling-resolution study; never a canonical calibration gate",
        }
    ]

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
        assert "importlib.invalidate_caches()" in source
        assert 'name.startswith("commguard.")' in source
        assert 'INSTALL_SOURCE = "auto"' in source
        assert "wheel_candidates" in source
        assert "archive_candidates" in source
        assert "PINNED_PUBLIC_COMMIT" in source
        assert "DEVELOPMENT_SMOKE_TEST" in source
        assert "EXPECTED_PACKAGE_SHA256" in source
        assert "EXPECTED_NOTEBOOK_SHA256" in source
        assert "PIP_FREEZE" in source
        assert "--no-build-isolation" in source
        assert "--no-deps" in source
        assert 'checkout", "main' not in source
        assert "@main" not in source
        assert "NEXT STEP:" in source
        assert "SHA-256" in source
        assert notebook["metadata"]["commguard"]["required_accelerator"] == "two NVIDIA T4 GPUs"
        assert any("parameters" in cell.get("metadata", {}).get("tags", []) for cell in code_cells)
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
    assert 'RUN_MODE = \\"smoke\\"' in adversarial
    assert 'RUN_FULL_PERIODIC_SYNCHRONIZATION_STUDY = RUN_MODE == \\"full\\"' in adversarial
    assert "ADVERSARIAL_HUMAN_APPROVAL = False" in adversarial

    calibration = (notebook_root / names[0]).read_text(encoding="utf-8")
    calibration_notebook = json.loads(calibration)
    calibration_source = "\n".join(
        "".join(cell["source"])
        for cell in calibration_notebook["cells"]
        if cell["cell_type"] == "code"
    )
    assert "(1, 4, 16, 64, 128)" in calibration_source
    assert "REPETITIONS = 1 if DEVELOPMENT_SMOKE_ONLY else 5" in calibration_source
    assert "SAMPLING_INTERVAL_S = 0.2" in calibration_source
    assert "intervals_s=(1.0, 0.5, 0.2)" in calibration_source
    assert "SCIENTIFIC_ACCEPTANCE_ELIGIBLE" in calibration_source
    assert "CALIBRATION_ACCEPTED" in calibration_source
    assert "SOURCE_CLEAN" in calibration_source
    assert "RESULT_SUPPORTED" in calibration_source
    assert "MODERN_CAPTURE_GATE_PASSED" in calibration_source
    assert "idle_usable_repetitions" in calibration_source
    assert "exact_calibration_reference_for_next_notebook" in calibration_source
    assert 'shutil.which("nvidia-smi")' in calibration_source
    assert "Expected exactly two Tesla T4 GPUs" in calibration_source
    assert "torch.distributed.is_nccl_available()" in calibration_source
    assert "memory_total_mib" in calibration_source
    assert calibration_source.count("NOTEBOOK_RUN_ID = datetime.now") == 1
    assert "commguard-calibration-v4-{NOTEBOOK_RUN_ID}" in calibration_source
    assert "ARTIFACTS.mkdir(parents=True, exist_ok=False)" in calibration_source
    assert "environment/notebook-bootstrap.json" in calibration_source
    assert '"source_repository": "commguard-source"' in calibration_source
    assert '"artifact_schema_version": CURRENT_SCHEMA_VERSION' in calibration_source
    assert "progress_callback=" in calibration_source
    assert "standard_sweep_validation" in calibration_source
    assert "export_with_checksum" in calibration_source
    assert "materialize_calibration_package" in calibration_source
    assert "FAILED/INCONCLUSIVE EVIDENCE WAS PRESERVED" in calibration_source
    assert "Do not run the benign notebook" in calibration_source
    calibration_ids = [cell["id"] for cell in calibration_notebook["cells"]]
    assert calibration_ids.index("cal-workspace") < calibration_ids.index("cal-context")

    benign = (notebook_root / names[1]).read_text(encoding="utf-8")
    assert "PRIOR_CALIBRATION_ARTIFACT_PATH" in benign
    assert "EXPECTED_PRIOR_CALIBRATION_SHA256" in benign
    assert "prior_session_calibration_reference" in benign
    assert "current_session_gating_calibration" in benign
    assert "prior_calibration_reference=PRIOR_CALIBRATION_REFERENCE" in benign
    assert "calibration_paths = sorted" not in benign

    detector = (notebook_root / names[2]).read_text(encoding="utf-8")
    assert "BENIGN_MATRIX_SUMMARY_PATH" in detector
    assert "BENIGN_EXTRACTION.calibration_reference" in detector
    assert 'glob("extraction-*' not in detector


def test_canonical_notebooks_match_their_generator() -> None:
    result = subprocess.run(
        [sys.executable, "tools/generate_canonical_notebooks.py", "--check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_active_notebook_directory_contains_only_reviewed_workflow() -> None:
    notebook_root = ROOT / "notebooks"
    inventory = json.loads((notebook_root / "canonical_notebooks.json").read_text(encoding="utf-8"))

    canonical_names = [entry["filename"] for entry in inventory["canonical_notebooks"]]
    historical_names = [entry["filename"] for entry in inventory.get("historical_notebooks", [])]

    reviewed_names = sorted([*canonical_names, *historical_names])
    present_names = sorted(path.name for path in notebook_root.glob("*.ipynb"))

    assert present_names == reviewed_names
    diagnostic = ROOT / "notebooks/diagnostics/commguard_calibration_v4_sampling_study.ipynb"
    notebook = json.loads(diagnostic.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert all(
        cell.get("execution_count") is None and not cell.get("outputs")
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )


def test_delivery_policy_scan_passes() -> None:
    result = subprocess.run(
        [sys.executable, "tools/verify_delivery.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_delivery_policy_rejects_tracked_personal_media_path(tmp_path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    leaked_path = "/" + "media/example/private/project.txt"
    (tmp_path / "public.md").write_text(f"local source: {leaked_path}\n", encoding="utf-8")
    subprocess.run(["git", "add", "public.md"], cwd=tmp_path, check=True)

    result = subprocess.run(
        [sys.executable, str(ROOT / "tools/verify_delivery.py"), "--root", str(tmp_path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "public.md contains prohibited personal path/link" in result.stdout


def test_agent_state_is_ignored_untracked_and_not_scanner_exempt() -> None:
    ignored = subprocess.run(
        ["git", "check-ignore", "-q", ".agent/state/private.md"],
        cwd=ROOT,
        check=False,
    )
    tracked = subprocess.run(
        ["git", "ls-files", ".agent/state"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    pending_deletions = subprocess.run(
        ["git", "diff", "--name-only", "--diff-filter=D"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    scanner = (ROOT / "tools/verify_delivery.py").read_text(encoding="utf-8")

    assert ignored.returncode == 0
    tracked_paths = set(tracked.stdout.splitlines())
    deleted_paths = set(pending_deletions.stdout.splitlines())
    assert not tracked_paths or tracked_paths <= deleted_paths
    assert ".agent/state" not in scanner


def test_core_install_has_no_forced_dependencies() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert project["project"]["name"] == "commguard"
    assert project["project"]["dependencies"] == []


def test_evidence_docs_preserve_the_negative_result_and_archive_hashes() -> None:
    evidence = (ROOT / "docs/evidence-index.md").read_text(encoding="utf-8")
    current = (ROOT / "docs/current-results.md").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    report = (ROOT / "reports/research-report-template.md").read_text(encoding="utf-8")
    expected_hashes = {
        "0d52977b0db40bc06bf29155332ebdd48e581d0de063fd6930db42c8500ed70f",
        "946184076a7fdf6c71b2dd1bcd60d98d34dd63a52e020d399ebe044d73c7d660",
        "2460a44a2100d024bef6176aadd85cf9c57e343e7d3301c2dfc321acd99de48a",
        "13f0122997ac4127a6d45c032d5a2806d57e60f107e85e004f17b5f3e4308cc6",
        "840fbb6fd8860c3768dc98968e4c5d0c9ad96db4b8b04fe97e228274e34dc07d",
    }

    assert all(digest in evidence for digest in expected_hashes)
    assert "DDP versus idle" in current
    assert "training recall was 0" in report
    assert "reliable training-versus-inference detection" in readme
    assert "@main" not in readme
    assert "100% training detection" not in readme.lower()
    assert "not executed" in report


def test_public_claim_boundaries_remain_explicit() -> None:
    current = (ROOT / "docs/current-results.md").read_text(encoding="utf-8")
    limitations = (ROOT / "docs/limitations.md").read_text(encoding="utf-8")
    adversarial = (ROOT / "docs/adversarial-research.md").read_text(encoding="utf-8")
    central = (ROOT / "docs/central-monitoring-design.md").read_text(encoding="utf-8")

    assert "DDP versus idle" in current
    assert "the only training test run was missed" in current
    assert "tiny DDP-versus-idle diagnostic" in current
    assert "No successful modern calibration" in limitations
    assert "correctly `not_supported`" in limitations
    assert "cannot support a conclusion about PCIe telemetry" in limitations
    assert "Physical multi-node validation" in limitations
    assert "pending" in limitations
    assert "reliable training-versus-inference detection" in limitations
    assert "No adversarial GPU workload or detector result was executed" in adversarial
    assert "reference implementation" in central


def test_local_markdown_links_resolve() -> None:
    markdown_paths = [
        ROOT / "README.md",
        ROOT / "CODEX_COMPLETION_REPORT.md",
        *sorted((ROOT / "docs").glob("*.md")),
    ]
    markdown_paths.extend(sorted((ROOT / "reports").glob("*.md")))
    markdown_paths.extend(sorted((ROOT / "delivery").glob("*.md")))
    for source_path in markdown_paths:
        source = source_path.read_text(encoding="utf-8")
        for target in re.findall(r"\[[^]]+\]\(([^)]+)\)", source):
            if target.startswith(("https://", "http://", "mailto:", "#")):
                continue
            relative = target.split("#", 1)[0]
            assert (source_path.parent / relative).resolve().exists(), (
                f"broken local link in {source_path.relative_to(ROOT)}: {target}"
            )


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
