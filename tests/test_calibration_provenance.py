from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from commguard import orchestrator
from commguard.artifacts import ArtifactStore
from commguard.calibration import (
    analyze_calibration,
    build_calibration_reference,
    validate_standard_calibration_result,
    verify_calibration_reference,
)
from commguard.exceptions import ArtifactExistsError, CalibrationError
from commguard.provenance import ProvenanceContext
from commguard.schemas import CURRENT_SCHEMA_VERSION, LEGACY_SCHEMA_VERSION, validate_artifact
from commguard.workloads import calibration_workload_name, get_workload


def _pcie_sample(monotonic_ns: int, value: float) -> SimpleNamespace:
    reading = SimpleNamespace(supported=True, value=value)
    return SimpleNamespace(
        monotonic_ns=monotonic_ns,
        fields={"pcie_tx_bytes_per_s": reading, "pcie_rx_bytes_per_s": reading},
    )


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


def test_standard_sweep_collects_idle_and_five_repeated_payloads(tmp_path, monkeypatch) -> None:
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
    progress_events: list[dict] = []

    def experiment(workload: str, **kwargs):
        overrides = dict(kwargs["overrides"])
        calls.append((workload, overrides))
        payload = int(overrides.get("payload_mib", 0))
        signal = 100_000.0 if workload == "calibration_idle" else payload * 2_000_000.0
        run_id = f"run-{len(calls):02d}"
        config = get_workload(workload)
        config.update(overrides)
        config.update({"mode": config["mode"], "workload_name": workload})
        run_directory = tmp_path / "runs" / run_id
        run_directory.mkdir(parents=True)
        (run_directory / "manifest.json").write_text(
            json.dumps({"workload_name": workload, "config": config}),
            encoding="utf-8",
        )
        if workload == "calibration_idle":
            for rank in (0, 1):
                events = [
                    {"event": "measurement_start", "monotonic_ns": 100, "details": {}},
                    {"event": "heartbeat", "monotonic_ns": 150, "details": {}},
                    {
                        "event": "measurement_interval",
                        "monotonic_ns": 200,
                        "details": {
                            "measurement_start_monotonic_ns": 100,
                            "measurement_end_monotonic_ns": 200,
                        },
                    },
                    {"event": "measurement_end", "monotonic_ns": 200, "details": {}},
                ]
                (run_directory / f"rank-{rank}.events.jsonl").write_text(
                    "\n".join(json.dumps(event) for event in events) + "\n",
                    encoding="utf-8",
                )
        return {
            "run_id": run_id,
            "manifest": {
                "participation_valid": True,
                "exit_status": "completed",
                "workload_name": workload,
                "config": config,
                "measured_duration_seconds": 30.0,
                "environment": {
                    "gpus": [{"uuid": "GPU-0"}, {"uuid": "GPU-1"}],
                },
            },
            "pcie_supported": True,
            "pcie_total_mean_bytes_per_s": signal,
            "pcie_total_median_bytes_per_s": signal,
            "pcie_sample_count": 10,
            "telemetry_diagnostics": {
                "requested_interval_s": 0.5,
                "mean_interval_s": 0.5,
            },
        }

    monkeypatch.setattr(orchestrator, "run_experiment", experiment)
    monkeypatch.setattr(
        orchestrator,
        "check_environment",
        lambda **kwargs: {"environment_fingerprint": "environment-dual-t4"},
    )

    result = orchestrator.run_calibration_sweep(
        output=tmp_path,
        provenance=context,
        progress_callback=progress_events.append,
    )

    assert result["status"] == "supported"
    assert len(calls) == 30
    assert [name for name, _ in calls].count("calibration_idle") == 5
    assert {name for name, _ in calls if name != "calibration_idle"} == {
        "collective_all_reduce_1mib",
        "collective_all_reduce_4mib",
        "collective_all_reduce_16mib",
        "collective_all_reduce_64mib",
        "collective_all_reduce_128mib",
    }
    assert {item["payload_mib"] for item in result["payload_summaries"]} == {
        1.0,
        4.0,
        16.0,
        64.0,
        128.0,
    }
    assert result["idle_baseline_usable_repetitions"] == 5
    assert result["reference"]["calibration_relationship"] == "current_session"
    saved = json.loads(Path(result["path"]).read_text(encoding="utf-8"))
    assert saved["schema_version"] == CURRENT_SCHEMA_VERSION
    assert saved["modern_capture_gate_passed"] is True
    assert result["standard_sweep_validation"]["planned_run_count"] == 30
    assert result["standard_sweep_validation"]["clean_standard_calibration"] is True
    assert len(list((tmp_path / "results/calibration-sweeps").glob("*/started.json"))) == 1
    assert len(list((tmp_path / "results/calibration-sweeps").glob("*/completed.json"))) == 1
    assert progress_events[0]["stage"] == "sweep_started"
    assert progress_events[-1]["stage"] == "sweep_completed"
    assert sum(event["stage"] == "run_started" for event in progress_events) == 30
    assert sum(event["stage"] == "run_completed" for event in progress_events) == 30

    with pytest.raises(ArtifactExistsError, match="completed calibration sweep"):
        orchestrator.run_calibration_sweep(output=tmp_path, provenance=context)

    first_idle = next(
        row for row in result["observations"] if row["observation_type"] == "idle_baseline"
    )
    event_path = tmp_path / first_idle["run_directory"] / "rank-0.events.jsonl"
    with event_path.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                {
                    "event": "collective_complete",
                    "monotonic_ns": 175,
                    "details": {"collective": "all_reduce"},
                }
            )
            + "\n"
        )
    with pytest.raises(CalibrationError, match="measured collective events"):
        validate_standard_calibration_result(tmp_path, result)


def test_payload_specific_calibration_identities_are_stable() -> None:
    names = [calibration_workload_name("all_reduce", payload) for payload in (1, 4, 16, 64)]

    assert names == [
        "collective_all_reduce_1mib",
        "collective_all_reduce_4mib",
        "collective_all_reduce_16mib",
        "collective_all_reduce_64mib",
    ]
    for payload, name in zip((1, 4, 16, 64), names, strict=True):
        config = get_workload(name)
        assert config["payload_mib"] == payload
        assert config["collective"] == "all_reduce"
        assert f"-{payload}mib-" in config["config_id"]


def test_run_rejects_payload_that_disagrees_with_calibration_name(tmp_path) -> None:
    with pytest.raises(ValueError, match="identity conflicts"):
        orchestrator.run_experiment(
            "collective_all_reduce_1mib",
            output=tmp_path,
            overrides={"payload_mib": 4},
        )
    assert not list(tmp_path.iterdir())


def test_partial_sweep_is_preserved_and_rejected_on_retry(tmp_path, monkeypatch) -> None:
    context = ProvenanceContext(
        experiment_session_id="session-partial",
        collection_id="collection-partial",
        corpus_id="corpus-partial",
        node_id="node-0",
        source_commit="a" * 40,
        source_dirty=False,
        notebook_version="calibration-v3",
        input_archive_sha256=None,
        random_seed=1337,
    )
    monkeypatch.setattr(
        orchestrator,
        "run_experiment",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("interrupted")),
    )

    with pytest.raises(RuntimeError, match="interrupted"):
        orchestrator.run_calibration_sweep(output=tmp_path, provenance=context)

    assert len(list((tmp_path / "results/calibration-sweeps").glob("*/started.json"))) == 1
    assert not list((tmp_path / "results/calibration-sweeps").glob("*/completed.json"))
    with pytest.raises(ArtifactExistsError, match="partial or in-progress"):
        orchestrator.run_calibration_sweep(output=tmp_path, provenance=context)


def test_sweep_is_allowed_in_a_new_output_directory(tmp_path, monkeypatch) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    context = ProvenanceContext(
        experiment_session_id="session-new-root",
        collection_id="collection-new-root",
        corpus_id="corpus-new-root",
        node_id="node-0",
        source_commit="a" * 40,
        source_dirty=False,
        notebook_version="calibration-v3",
        input_archive_sha256=None,
        random_seed=1337,
    )
    monkeypatch.setattr(
        orchestrator,
        "run_experiment",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("stop after marker")),
    )
    for output in (first, second):
        with pytest.raises(RuntimeError, match="stop after marker"):
            orchestrator.run_calibration_sweep(output=output, provenance=context)
        assert len(list((output / "results/calibration-sweeps").glob("*/started.json"))) == 1


def test_different_context_cannot_append_sweep_to_same_output(tmp_path) -> None:
    marker = tmp_path / "results/calibration-sweeps/previous/started.json"
    marker.parent.mkdir(parents=True)
    marker.write_text("{}\n", encoding="utf-8")
    context = ProvenanceContext.create(corpus_id="corpus-new-context")

    with pytest.raises(ArtifactExistsError, match="calibration sweep"):
        orchestrator.run_calibration_sweep(output=tmp_path, provenance=context)


def test_idle_pcie_observation_uses_common_measured_interval(tmp_path) -> None:
    run_directory = tmp_path / "run"
    run_directory.mkdir()
    intervals = ((100, 200), (110, 190))
    for rank, (start, end) in enumerate(intervals):
        event = {
            "event": "measurement_interval",
            "monotonic_ns": end,
            "details": {
                "measurement_start_monotonic_ns": start,
                "measurement_end_monotonic_ns": end,
            },
        }
        (run_directory / f"rank-{rank}.events.jsonl").write_text(
            json.dumps(event) + "\n", encoding="utf-8"
        )
    samples = [
        _pcie_sample(90, 1000.0),
        _pcie_sample(120, 10.0),
        _pcie_sample(180, 20.0),
        _pcie_sample(210, 1000.0),
    ]

    result = orchestrator._pcie_observation(samples, run_directory)

    assert result["measurement_phase_bounded"] is True
    assert result["pcie_sample_count"] == 2
    assert result["pcie_total_mean_bytes_per_s"] == 30.0
