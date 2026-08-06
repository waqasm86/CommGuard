from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def notebook_source(name: str) -> str:
    notebook = json.loads((ROOT / "notebooks" / name).read_text(encoding="utf-8"))
    return "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])


def test_calibration_v4_confirmatory_configuration() -> None:
    source = notebook_source("commguard_calibration_v4.ipynb")

    assert 'RUN_MODE = "smoke"' in source
    assert 'DEVELOPMENT_SMOKE_ONLY = RUN_MODE == "smoke"' in source
    assert "PAYLOADS = (16,) if DEVELOPMENT_SMOKE_ONLY else (1, 4, 16, 64, 128)" in source
    assert "REPETITIONS = 1 if DEVELOPMENT_SMOKE_ONLY else 5" in source
    assert "SAMPLING_INTERVAL_S = 0.2" in source
    assert "total_runs" in source
    assert "run_calibration_sweep" in source


def test_calibration_v4_preserves_failed_evidence_before_raising() -> None:
    source = notebook_source("commguard_calibration_v4.ipynb")

    export_position = source.index("export_with_checksum")
    gate_position = source.index("elif not CALIBRATION_ACCEPTED")

    assert export_position < gate_position
    assert "Do not run the benign notebook" in source


def test_calibration_v4_requires_all_downstream_conditions() -> None:
    source = notebook_source("commguard_calibration_v4.ipynb")

    assert "SCIENTIFIC_ACCEPTANCE_ELIGIBLE = not DEVELOPMENT_SMOKE_ONLY" in source
    assert "SOURCE_CLEAN = not SOURCE_DIRTY" in source
    assert 'RESULT_SUPPORTED = CALIBRATION["result_state"] == "supported"' in source
    assert "MODERN_CAPTURE_GATE_PASSED" in source
    assert "and RESULT_SUPPORTED" in source
    assert "and MODERN_CAPTURE_GATE_PASSED" in source
    assert "and SOURCE_CLEAN" in source


def test_calibration_v3_remains_historical_and_unexecuted() -> None:
    inventory = json.loads(
        (ROOT / "notebooks/canonical_notebooks.json").read_text(encoding="utf-8")
    )

    assert inventory["canonical_notebooks"][0]["filename"] == ("commguard_calibration_v4.ipynb")
    assert inventory["historical_notebooks"][0]["filename"] == ("commguard_calibration_v3.ipynb")

    historical = json.loads(
        (ROOT / "notebooks/commguard_calibration_v3.ipynb").read_text(encoding="utf-8")
    )

    for cell in historical["cells"]:
        if cell["cell_type"] == "code":
            assert cell.get("execution_count") is None
            assert not cell.get("outputs")
