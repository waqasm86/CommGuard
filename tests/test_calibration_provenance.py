from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from commguard import orchestrator
from commguard.artifacts import ArtifactStore
from commguard.calibration import (
    analyze_calibration,
    build_calibration_reference,
    verify_calibration_reference,
)
from commguard.exceptions import CalibrationError
from commguard.provenance import ProvenanceContext
from commguard.schemas import CURRENT_SCHEMA_VERSION, LEGACY_SCHEMA_VERSION, validate_artifact


def modern_calibration(
    *,
    session_id: str = "session-current",
    collection_id: str = "collection-current",
    environment_fingerprint: str = "environment-dual-t4",
    source_commit: str = "a" * 40,
) -> dict:
    rows = [
        {
            "observation_type": "idle_baseline",
            "payload_mib": 0,
            "pcie_total_mean_bytes_per_s": value,
            "pcie_supported": True,
            "participation_valid": True,
        }
        for value in (100_000, 110_000, 120_000)
    ]
    for payload, center in ((1, 2_000_000), (4, 4_000_000), (16, 8_000_000), (64, 16_000_000)):
        rows.extend(
            {
                "observation_type": "collective",
                "payload_mib": payload,
                "pcie_total_mean_bytes_per_s": center + offset,
                "pcie_supported": True,
                "participation_valid": True,
            }
            for offset in (-10_000, 0, 10_000)
        )
    result = analyze_calibration(rows)
    result.update(
        {
            "experiment_session_id": session_id,
            "collection_id": collection_id,
            "corpus_id": "corpus-calibration",
            "node_id": "node-0",
            "environment_fingerprint": environment_fingerprint,
            "source_commit": source_commit,
            "source_dirty": False,
        }
    )
    validate_artifact(result)
    return result


def save_calibration(
    root: Path,
    name: str,
    calibration: dict,
    *,
    current_session_id: str = "session-current",
) -> tuple[Path, dict]:
    store = ArtifactStore(root)
    store.initialize()
    path = store.write_json(f"results/{name}.json", calibration)
    reference = build_calibration_reference(
        root,
        path,
        current_experiment_session_id=current_session_id,
    )
    return path, reference


def test_exact_reference_selects_one_of_multiple_calibrations(tmp_path) -> None:
    selected, reference = save_calibration(tmp_path, "calibration-a", modern_calibration())
    save_calibration(
        tmp_path,
        "calibration-z",
        modern_calibration(session_id="session-other", collection_id="collection-other"),
    )

    resolved, loaded = verify_calibration_reference(
        tmp_path,
        reference,
        expected_experiment_session_ids={"session-current"},
        expected_environment_fingerprints={"environment-dual-t4"},
        expected_source_commits={"a" * 40},
    )

    assert resolved == selected
    assert loaded["experiment_session_id"] == "session-current"


def test_reference_rejects_wrong_hash_and_missing_artifact(tmp_path) -> None:
    _, reference = save_calibration(tmp_path, "calibration-current", modern_calibration())
    wrong_hash = {**reference, "calibration_sha256": "0" * 64}
    missing = {**reference, "calibration_artifact_path": "results/calibration-missing.json"}

    with pytest.raises(CalibrationError, match="SHA-256 mismatch"):
        verify_calibration_reference(tmp_path, wrong_hash)
    with pytest.raises(CalibrationError, match="is missing"):
        verify_calibration_reference(tmp_path, missing)


@pytest.mark.parametrize(
    ("keyword", "expected", "message"),
    [
        ("expected_experiment_session_ids", {"session-wrong"}, "experiment session mismatch"),
        (
            "expected_environment_fingerprints",
            {"environment-wrong"},
            "environment fingerprint mismatch",
        ),
        ("expected_source_commits", {"b" * 40}, "source commit mismatch"),
    ],
)
def test_reference_rejects_incompatible_provenance(
    tmp_path,
    keyword: str,
    expected: set[str],
    message: str,
) -> None:
    _, reference = save_calibration(tmp_path, "calibration-current", modern_calibration())

    with pytest.raises(CalibrationError, match=message):
        verify_calibration_reference(tmp_path, reference, **{keyword: expected})


def test_reference_rejects_legacy_schema_for_modern_gate(tmp_path) -> None:
    legacy = {
        "artifact_kind": "calibration_result",
        "schema_version": LEGACY_SCHEMA_VERSION,
        "status": "supported",
        "observations": [],
        "falsification_reasons": [],
        "experiment_session_id": "session-legacy",
        "collection_id": "collection-legacy",
        "environment_fingerprint": "environment-legacy",
        "source_commit": "1" * 40,
    }
    _, reference = save_calibration(
        tmp_path,
        "calibration-legacy",
        legacy,
        current_session_id="session-legacy",
    )

    with pytest.raises(CalibrationError, match="modern evaluation requires calibration schema"):
        verify_calibration_reference(tmp_path, reference)
    _, loaded = verify_calibration_reference(tmp_path, reference, allow_legacy=True)
    assert loaded["schema_version"] == LEGACY_SCHEMA_VERSION


def test_prior_and_current_session_labels_are_not_interchangeable(tmp_path) -> None:
    calibration = modern_calibration()
    _, current = save_calibration(tmp_path, "calibration-current", calibration)
    prior = build_calibration_reference(
        tmp_path,
        current["calibration_artifact_path"],
        current_experiment_session_id="session-next",
    )

    assert current["calibration_relationship"] == "current_session"
    assert current["calibration_created_in_current_session"] is True
    assert prior["calibration_relationship"] == "prior_session"
    assert prior["calibration_created_in_current_session"] is False
    with pytest.raises(CalibrationError, match="current-session calibration"):
        verify_calibration_reference(tmp_path, prior)
    verify_calibration_reference(tmp_path, prior, require_current_session=False)


def test_reference_rejects_reused_path_with_altered_content(tmp_path) -> None:
    path, reference = save_calibration(tmp_path, "calibration-current", modern_calibration())
    changed = copy.deepcopy(modern_calibration())
    changed["source_commit"] = "b" * 40
    path.write_text(json.dumps(changed), encoding="utf-8")

    with pytest.raises(CalibrationError, match="SHA-256 mismatch"):
        verify_calibration_reference(tmp_path, reference)


def test_standard_sweep_collects_idle_and_four_repeated_payloads(tmp_path, monkeypatch) -> None:
    context = ProvenanceContext(
        experiment_session_id="session-current",
        collection_id="collection-current",
        corpus_id="corpus-current",
        node_id="node-0",
        source_commit="a" * 40,
        source_dirty=False,
        notebook_version="calibration-v3",
        input_archive_sha256=None,
        random_seed=1337,
    )
    calls: list[tuple[str, dict]] = []

    def experiment(workload: str, **kwargs):
        overrides = dict(kwargs["overrides"])
        calls.append((workload, overrides))
        payload = int(overrides.get("payload_mib", 0))
        signal = 100_000.0 if workload == "calibration_idle" else payload * 2_000_000.0
        return {
            "run_id": f"run-{len(calls):02d}",
            "manifest": {"participation_valid": True, "exit_status": "completed"},
            "pcie_supported": True,
            "pcie_total_mean_bytes_per_s": signal,
            "pcie_total_median_bytes_per_s": signal,
            "pcie_sample_count": 10,
        }

    monkeypatch.setattr(orchestrator, "run_experiment", experiment)
    monkeypatch.setattr(
        orchestrator,
        "check_environment",
        lambda **kwargs: {"environment_fingerprint": "environment-dual-t4"},
    )

    result = orchestrator.run_calibration_sweep(output=tmp_path, provenance=context)

    assert result["status"] == "supported"
    assert len(calls) == 15
    assert [name for name, _ in calls].count("calibration_idle") == 3
    assert {item["payload_mib"] for item in result["payload_summaries"]} == {
        1.0,
        4.0,
        16.0,
        64.0,
    }
    assert result["idle_baseline_usable_repetitions"] == 3
    assert result["reference"]["calibration_relationship"] == "current_session"
    saved = json.loads(Path(result["path"]).read_text(encoding="utf-8"))
    assert saved["schema_version"] == CURRENT_SCHEMA_VERSION
    assert saved["modern_capture_gate_passed"] is True
