"""Materialize reviewable stage packages from real CommGuard evidence."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from commguard.artifacts.storage import ArtifactStore, sha256_file
from commguard.scope import PROTOTYPE_SCOPE_DECLARATION, with_prototype_scope

REPOSITORY_URL = "https://github.com/waqasm86/CommGuard"


def _load_json(root: Path, relative: str | Path) -> dict[str, Any]:
    path = (root / relative).resolve()
    # Ensure path is relative to root for security
    _ = path.relative_to(root.resolve())
    with open(path, encoding='utf-8') as f:
        result: dict[str, Any] = json.load(f)
        return result


def _load_optional_json(root: Path, relative: str | Path) -> dict[str, Any] | None:
    path = (root / relative).resolve()
    try:
        _ = path.relative_to(root.resolve())
    except ValueError:
        return None
    if not path.exists():
        return None
    with open(path, encoding='utf-8') as f:
        result: dict[str, Any] = json.load(f)
        return result


def configuration_hash(configuration: Mapping[str, Any]) -> str:
    """Hash a deterministic JSON configuration."""
    encoded = json.dumps(
        configuration,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _jsonl(store: ArtifactStore, path: Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    return store.write_jsonl(path, rows, validate=False)


def _csv(store: ArtifactStore, path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    columns = sorted({str(key) for row in rows for key in row})
    target = store.resolve(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite artifact: {target}")
    with target.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    column: (
                        json.dumps(row.get(column), sort_keys=True)
                        if isinstance(row.get(column), (dict, list, tuple))
                        else row.get(column)
                    )
                    for column in columns
                }
            )
    return target


def _common_metadata(
    *,
    workflow: str,
    provenance: Mapping[str, Any],
    notebook_filename: str,
    notebook_sha256: str,
    configuration: Mapping[str, Any],
) -> dict[str, Any]:
    return with_prototype_scope(
        {
            "workflow": workflow,
            "repository_url": REPOSITORY_URL,
            "git_commit": provenance.get("source_commit"),
            "git_dirty": provenance.get("source_dirty"),
            "commguard_version": provenance.get("commguard_version", "0.2.0"),
            "notebook_filename": notebook_filename,
            "notebook_sha256": notebook_sha256,
            "configuration_hash": configuration_hash(configuration),
            "experiment_session_id": provenance.get("experiment_session_id"),
            "corpus_id": provenance.get("corpus_id"),
            "limitations": [PROTOTYPE_SCOPE_DECLARATION],
        }
    )


def _write_common(
    store: ArtifactStore,
    prefix: Path,
    *,
    environment: Mapping[str, Any],
    provenance: Mapping[str, Any],
    metadata: Mapping[str, Any],
    status: Mapping[str, Any],
) -> None:
    store.write_json(prefix / "environment.json", with_prototype_scope(environment), validate=False)
    store.write_json(
        prefix / "provenance.json",
        with_prototype_scope({**provenance, **metadata}),
        validate=False,
    )
    store.write_json(
        prefix / "run_status.json",
        with_prototype_scope({**metadata, **status}),
        validate=False,
    )


def _checksums(store: ArtifactStore, prefix: Path) -> Path:
    directory = store.resolve(prefix)
    lines = [
        f"{sha256_file(path)}  {path.relative_to(directory).as_posix()}"
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.name != "sha256sums.txt"
    ]
    return store.write_text(prefix / "sha256sums.txt", "\n".join(lines) + "\n")


def _plot_calibration(path: Path, observations: Sequence[Mapping[str, Any]]) -> None:
    collectives = [
        row
        for row in observations
        if row.get("observation_type") == "collective"
        and isinstance(row.get("pcie_total_mean_bytes_per_s"), (int, float))
    ]
    if not collectives:
        raise ValueError("cannot generate calibration_plot.png without actual collective data")
    import matplotlib.pyplot as plt

    grouped: dict[float, list[float]] = defaultdict(list)
    for row in collectives:
        grouped[float(row["payload_mib"])].append(float(row["pcie_total_mean_bytes_per_s"]))
    payloads = sorted(grouped)
    values = [statistics.median(grouped[payload]) for payload in payloads]
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(payloads, values, marker="o")
    axis.set(xlabel="AllReduce payload (MiB)", ylabel="Median NVML PCIe RX+TX (bytes/s)")
    axis.set_title("Observed calibration response")
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def materialize_calibration_package(
    artifact_root: str | Path,
    calibration: Mapping[str, Any],
    sampling_comparison: Sequence[Mapping[str, Any]],
    *,
    environment: Mapping[str, Any],
    provenance: Mapping[str, Any],
    notebook_filename: str,
    notebook_sha256: str,
    configuration: Mapping[str, Any],
    development_smoke_only: bool,
) -> Path:
    """Write the required calibration review files from measured evidence."""
    store = ArtifactStore(artifact_root)
    store.initialize()
    prefix = Path("prototype/calibration")
    metadata = _common_metadata(
        workflow="calibration",
        provenance=provenance,
        notebook_filename=notebook_filename,
        notebook_sha256=notebook_sha256,
        configuration=configuration,
    )
    observations = [dict(row) for row in calibration.get("observations", [])]
    accepted = (
        calibration.get("result_state") == "supported"
        and calibration.get("modern_capture_gate_passed") is True
        and not development_smoke_only
    )
    manifest = with_prototype_scope(
        {
            **metadata,
            "configuration": dict(configuration),
            "development_smoke_only": development_smoke_only,
            "scientific_acceptance_eligible": not development_smoke_only,
            "target_sampling_interval_s": calibration.get("target_sampling_interval_s"),
            "raw_unit_metadata": {
                "pcie_nvml_raw_unit": "KB/second (NVML API)",
                "pcie_bytes_multiplier": 1024,
                "pcie_conversion_convention": "1 NVML KB = 1024 bytes",
                "pcie_counter_query_window_ms": 20,
                "stored_pcie_unit": "bytes/second",
            },
        }
    )
    store.write_json(prefix / "calibration_manifest.json", manifest, validate=False)
    _jsonl(store, prefix / "calibration_runs.jsonl", observations)
    store.write_json(prefix / "calibration_summary.json", calibration, validate=False)
    comparison_rows = [dict(row) for row in sampling_comparison]
    _csv(store, prefix / "sampling_interval_comparison.csv", comparison_rows)
    store.write_json(
        prefix / "sampling_interval_comparison.json",
        {"intervals": comparison_rows},
        validate=False,
    )
    acceptance = {
        "accepted": accepted,
        "result_state": calibration.get("result_state", calibration.get("status")),
        "launch_completed": all(row.get("exit_status") == "completed" for row in observations),
        "scientific_acceptance_eligible": not development_smoke_only,
        "development_smoke_only": development_smoke_only,
        "reasons": list(calibration.get("falsification_reasons", [])),
        "accepted_payload_range_mib": calibration.get("supported_payload_range_mib", []),
    }
    store.write_json(prefix / "calibration_acceptance.json", acceptance, validate=False)
    plot_path = store.resolve(prefix / "calibration_plot.png")
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    _plot_calibration(plot_path, observations)
    _write_common(
        store,
        prefix,
        environment=environment,
        provenance=provenance,
        metadata=metadata,
        status={
            "execution_state": "completed",
            "scientific_acceptance_state": "accepted" if accepted else "not_accepted",
            **acceptance,
        },
    )
    _checksums(store, prefix)
    return store.resolve(prefix)


def _load_run_json(root: Path, relative: str | Path) -> dict[str, Any]:
    """Helper to load JSON from a run directory."""
    path = (root / relative).resolve()
    path.relative_to(root.resolve())
    with open(path, encoding='utf-8') as f:
        result: dict[str, Any] = json.load(f)
        return result


def _aggregate_features(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["run_id"])].append(row)
    output: list[dict[str, Any]] = []
    for _run_id, records in sorted(grouped.items()):
        first = records[0]
        aggregate: dict[str, Any] = {
            key: first.get(key)
            for key in (
                "run_id",
                "experiment_session_id",
                "corpus_id",
                "workload_family",
                "workload_config_id",
                "target_label",
                "designation",
            )
        }
        aggregate["source_window_count"] = len(records)
        for key in sorted(set().union(*(record.keys() for record in records))):
            values = [
                float(record[key])
                for record in records
                if isinstance(record.get(key), (int, float))
                and not isinstance(record.get(key), bool)
                and math.isfinite(float(record[key]))
            ]
            if values and key not in aggregate:
                aggregate[key] = statistics.fmean(values)
        output.append(aggregate)
    return output


def materialize_corpus_package(
    artifact_root: str | Path,
    matrix: Mapping[str, Any],
    *,
    environment: Mapping[str, Any],
    provenance: Mapping[str, Any],
    notebook_filename: str,
    notebook_sha256: str,
    configuration: Mapping[str, Any],
    development_smoke_only: bool,
) -> Path:
    """Write corpus, completion, and complete-run feature deliverables."""
    root = Path(artifact_root).resolve()
    store = ArtifactStore(root)
    store.initialize()
    prefix = Path("prototype/corpus")
    metadata = _common_metadata(
        workflow="benign_corpus",
        provenance=provenance,
        notebook_filename=notebook_filename,
        notebook_sha256=notebook_sha256,
        configuration=configuration,
    )
    planned = _load_run_json(root, str(matrix["planned_corpus_manifest"]))
    final = _load_run_json(root, str(matrix["final_corpus_manifest"]))
    planned_rows = [dict(row) for row in planned["planned_runs"]]
    completion_rows = [
        {
            "plan_id": row["plan_id"],
            "workload_family": row["workload_family"],
            "configuration_id": row.get("workload_config_id"),
            "repetition": row.get("config", {}).get("repetition"),
            "accepted_run_id": row.get("accepted_run_id"),
            "completed": row.get("accepted_run_id") is not None,
        }
        for row in final["planned_runs"]
    ]
    store.write_json(prefix / "corpus_manifest.json", with_prototype_scope(final), validate=False)
    _csv(store, prefix / "planned_run_matrix.csv", planned_rows)
    store.write_json(prefix / "planned_run_matrix.json", {"runs": planned_rows}, validate=False)
    run_manifests = [
        _load_run_json(root, Path("runs") / run_id / "manifest.json")
        for run_id in matrix.get("run_ids", [])
    ]
    _jsonl(store, prefix / "corpus_runs.jsonl", run_manifests)
    acceptance = {
        "accepted": bool(matrix.get("primary_coverage_gate", {}).get("passed"))
        and not development_smoke_only,
        "coverage_gate": matrix.get("primary_coverage_gate"),
        "launch_completed_count": matrix.get("completed"),
        "launch_failed_count": matrix.get("failed"),
        "scientific_acceptance_eligible": not development_smoke_only,
        "development_smoke_only": development_smoke_only,
    }
    store.write_json(prefix / "corpus_acceptance.json", acceptance, validate=False)
    _csv(store, prefix / "completion_matrix.csv", completion_rows)
    store.write_json(prefix / "completion_matrix.json", {"runs": completion_rows}, validate=False)
    extraction = _load_run_json(root, str(matrix["feature_extraction_summary"]))
    feature_rows = [
        json.loads(line)
        for line in (root / extraction["feature_artifact"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    run_features = _aggregate_features(feature_rows)
    _csv(store, prefix / "run_features.csv", run_features)
    _jsonl(store, prefix / "run_features.jsonl", run_features)
    feature_columns = (
        sorted(set().union(*(row.keys() for row in run_features))) if run_features else []
    )
    communication = [column for column in feature_columns if "pcie_" in column]
    auxiliary = [
        column
        for column in feature_columns
        if column not in communication
        and any(
            token in column for token in ("utilization", "power", "clock", "temperature", "memory_")
        )
    ]
    store.write_json(
        prefix / "feature_schema.json",
        {
            "unit_of_analysis": "complete_run",
            "feature_sets": {
                "communication_only": communication,
                "auxiliary_only": auxiliary,
                "communication_plus_auxiliary": sorted(set(communication + auxiliary)),
            },
        },
        validate=False,
    )
    store.write_json(
        prefix / "feature_quality_report.json",
        {
            "run_feature_count": len(run_features),
            "source_window_feature_count": len(feature_rows),
            "coverage": extraction,
            "constant_and_insufficient_series_policy": "finite value or explicit null; no NaN",
        },
        validate=False,
    )
    _write_common(
        store,
        prefix,
        environment=environment,
        provenance=provenance,
        metadata=metadata,
        status={
            "execution_state": "completed",
            "scientific_acceptance_state": (
                "accepted" if acceptance["accepted"] else "not_accepted"
            ),
            **acceptance,
        },
    )
    _checksums(store, prefix)
    return store.resolve(prefix)


def _metric_rows(evaluation: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    feature_sets = {
        "communication_only": evaluation["primary_communication_only"]["metrics"],
        **evaluation.get("ablations", {}),
    }
    for feature_set, result in feature_sets.items():
        if not isinstance(result, Mapping) or "candidate_models" not in result:
            continue
        for model, blocks in result["candidate_models"].items():
            metrics = blocks.get("test", {})
            rows.append({"feature_set": feature_set, "model": model, **metrics})
    return rows


def materialize_detector_package(
    artifact_root: str | Path,
    evaluation: Mapping[str, Any],
    *,
    environment: Mapping[str, Any],
    provenance: Mapping[str, Any],
    notebook_filename: str,
    notebook_sha256: str,
    configuration: Mapping[str, Any],
) -> Path:
    """Write interpretable detector outputs without serializing unsafe models."""
    development_smoke_only = bool(configuration.get("development_smoke_only", False))
    store = ArtifactStore(artifact_root)
    store.initialize()
    prefix = Path("prototype/detector")
    metadata = _common_metadata(
        workflow="detector_evaluation",
        provenance=provenance,
        notebook_filename=notebook_filename,
        notebook_sha256=notebook_sha256,
        configuration=configuration,
    )
    store.write_json(
        prefix / "detector_manifest.json",
        with_prototype_scope({**metadata, "configuration": dict(configuration)}),
        validate=False,
    )
    store.write_json(prefix / "detector_metrics.json", evaluation, validate=False)
    metric_rows = _metric_rows(evaluation)
    _csv(store, prefix / "detector_metrics.csv", metric_rows)
    primary = evaluation["primary_communication_only"]["metrics"]
    family_rows = [
        {"workload_family": family, **metrics}
        for family, metrics in primary.get("per_family", {}).items()
    ]
    session_rows = [
        {"experiment_session_id": session, **metrics}
        for session, metrics in primary.get("per_session", {}).items()
    ]
    _csv(store, prefix / "per_family_metrics.csv", family_rows)
    _csv(store, prefix / "per_session_metrics.csv", session_rows)
    _csv(store, prefix / "predictions.csv", list(primary.get("run_predictions", [])))
    _csv(store, prefix / "feature_ablation.csv", metric_rows)
    store.write_json(
        prefix / "bootstrap_intervals.json",
        primary.get("run_level", {}).get("balanced_accuracy_bootstrap", {}),
        validate=False,
    )
    matrix = primary.get("run_level", {}).get("confusion_matrix_labels_0_nontraining_1_training")
    if matrix is None:
        raise ValueError("actual run-level confusion matrix is required")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(4, 4))
    image = axis.imshow(matrix, cmap="Blues")
    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            axis.text(column_index, row_index, str(value), ha="center", va="center")
    axis.set(
        xticks=(0, 1),
        yticks=(0, 1),
        xticklabels=("inference", "training"),
        yticklabels=("inference", "training"),
        xlabel="Predicted",
        ylabel="Actual",
        title="Run-level communication-only confusion matrix",
    )
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(store.resolve(prefix / "confusion_matrix.png"), dpi=150)
    plt.close(figure)
    model_card = (
        "# CommGuard prototype model card\n\n"
        "Primary task: training versus inference using complete-run communication-only "
        "features. Controls are excluded from the primary binary label and reported "
        "separately. Models are fitted only on grouped benign partitions; adversarial "
        "holdouts are not used for fitting or selection.\n\n"
        f"{PROTOTYPE_SCOPE_DECLARATION}\n\n"
        "No pickle is required for review: thresholds, selected model names, feature names, "
        "metrics, and predictions are stored in JSON/CSV. Refit from the pinned corpus to "
        "reproduce an sklearn estimator.\n"
    )
    store.write_text(prefix / "model_card.md", model_card)
    _write_common(
        store,
        prefix,
        environment=environment,
        provenance=provenance,
        metadata=metadata,
        status={
            "execution_state": "completed",
            "scientific_acceptance_state": (
                "development_smoke_only"
                if development_smoke_only
                else "reported_without_hard_coded_success"
            ),
            "development_smoke_only": development_smoke_only,
            "scientific_acceptance_eligible": not development_smoke_only,
            "coverage_gate": evaluation.get("coverage_gate"),
            "warnings": evaluation.get("warnings", []),
        },
    )
    _checksums(store, prefix)
    return store.resolve(prefix)


def materialize_adversarial_package(
    artifact_root: str | Path,
    matrix: Mapping[str, Any],
    heldout_evaluation: Mapping[str, Any],
    *,
    environment: Mapping[str, Any],
    provenance: Mapping[str, Any],
    notebook_filename: str,
    notebook_sha256: str,
    configuration: Mapping[str, Any],
) -> Path:
    """Write bounded periodic-synchronization tradeoff evidence from actual runs."""
    root = Path(artifact_root).resolve()
    store = ArtifactStore(root)
    store.initialize()
    prefix = Path("prototype/adversarial")
    metadata = _common_metadata(
        workflow="adversarial_redteam",
        provenance=provenance,
        notebook_filename=notebook_filename,
        notebook_sha256=notebook_sha256,
        configuration=configuration,
    )
    store.write_json(
        prefix / "adversarial_manifest.json",
        with_prototype_scope({**metadata, **dict(matrix), "bounded_research_redteam": True}),
        validate=False,
    )
    manifests = [
        _load_run_json(root, Path("runs") / run_id / "manifest.json")
        for run_id in matrix.get("run_ids", [])
    ]
    _jsonl(store, prefix / "adversarial_runs.jsonl", manifests)
    family = heldout_evaluation.get("heldout_adversarial_families", {}).get(
        "periodic_local_sgd", {}
    )
    by_configuration = family.get("by_configuration", {})
    extraction = _load_run_json(root, str(matrix["feature_extraction_summary"]))
    feature_rows = [
        json.loads(line)
        for line in (root / extraction["feature_artifact"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    features_by_run = {row["run_id"]: row for row in _aggregate_features(feature_rows)}

    def mean_gpu_feature(run_features: Mapping[str, Any], suffix: str) -> float | None:
        values = [
            float(run_features[key])
            for key in (f"gpu0__pcie_total__{suffix}", f"gpu1__pcie_total__{suffix}")
            if isinstance(run_features.get(key), (int, float))
            and math.isfinite(float(run_features[key]))
        ]
        return statistics.fmean(values) if values else None

    tradeoff_rows: list[dict[str, Any]] = []
    for manifest in manifests:
        config = dict(manifest.get("config", {}))
        config_id = str(config.get("config_id", ""))
        rank_evidence = dict(config.get("rank_runtime_evidence", {})).get("0", {})
        strategy = dict(rank_evidence.get("strategy_summary", {}))
        scored = dict(by_configuration.get(config_id, {}))
        run_features = features_by_run.get(str(manifest.get("run_id")), {})
        tradeoff_rows.append(
            {
                "run_id": manifest.get("run_id"),
                "configuration_id": config_id,
                "synchronization_interval_k": config.get("local_steps"),
                "detector_score": scored.get("detector_score_mean"),
                "predicted_label": scored.get("predicted_label"),
                "abstention": scored.get("abstention"),
                "actual_sync_rounds": strategy.get("actual_sync_rounds"),
                "final_loss": strategy.get("final_loss_proxy"),
                "loss_trajectory": strategy.get("loss_trajectory"),
                "steps_per_second": strategy.get("steps_per_second"),
                "tokens_per_second": strategy.get("throughput_tokens_per_s"),
                "parameter_state_agreement": strategy.get("parameter_state_agreement"),
                "parameter_divergence": strategy.get("pre_final_parameter_divergence"),
                "communication_bytes_proxy": strategy.get("communication_bytes_proxy"),
                "communication_mean_bytes_per_s": mean_gpu_feature(run_features, "mean"),
                "communication_duty_cycle": mean_gpu_feature(
                    run_features, "communication_duty_cycle"
                ),
                "burst_count": mean_gpu_feature(run_features, "burst_count"),
                "peak_communication_bytes_per_s": mean_gpu_feature(run_features, "maximum"),
            }
        )
    baseline = next(
        (row for row in tradeoff_rows if row.get("synchronization_interval_k") == 1), None
    )
    if baseline is None:
        raise ValueError("the periodic synchronization tradeoff requires a measured k=1 baseline")

    def relative_reduction(value: Any, baseline_value: Any) -> float | None:
        if not isinstance(value, (int, float)) or not isinstance(baseline_value, (int, float)):
            return None
        if not math.isfinite(float(value)) or not math.isfinite(float(baseline_value)):
            return None
        return 1.0 - float(value) / float(baseline_value) if baseline_value else None

    for row in tradeoff_rows:
        row["relative_communication_reduction"] = relative_reduction(
            row.get("communication_mean_bytes_per_s"),
            baseline.get("communication_mean_bytes_per_s"),
        )
        row["relative_communication_proxy_reduction"] = relative_reduction(
            row.get("communication_bytes_proxy"), baseline.get("communication_bytes_proxy")
        )
        row["throughput_utility_degradation"] = relative_reduction(
            row.get("tokens_per_second"), baseline.get("tokens_per_second")
        )
        row["final_loss_change_vs_k1"] = (
            float(row["final_loss"]) - float(baseline["final_loss"])
            if isinstance(row.get("final_loss"), (int, float))
            and isinstance(baseline.get("final_loss"), (int, float))
            else None
        )
    _csv(store, prefix / "adversarial_metrics.csv", tradeoff_rows)
    _csv(store, prefix / "utility_evasion_tradeoff.csv", tradeoff_rows)
    store.write_json(prefix / "heldout_evaluation.json", heldout_evaluation, validate=False)
    plottable = [
        row
        for row in tradeoff_rows
        if isinstance(row.get("synchronization_interval_k"), (int, float))
        and isinstance(row.get("detector_score"), (int, float))
    ]
    if not plottable:
        raise ValueError("actual detector scores are required for the tradeoff plot")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(
        [row["synchronization_interval_k"] for row in plottable],
        [row["detector_score"] for row in plottable],
        marker="o",
    )
    axis.set(xlabel="Synchronization interval k", ylabel="Frozen detector score")
    axis.set_title("Periodic synchronization utility/evasion tradeoff")
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(store.resolve(prefix / "utility_evasion_tradeoff.png"), dpi=150)
    plt.close(figure)
    _write_common(
        store,
        prefix,
        environment=environment,
        provenance=provenance,
        metadata=metadata,
        status={
            "execution_state": "completed",
            "scientific_acceptance_state": "reported_without_evasion_success_claim",
            "bounded_research_redteam": True,
        },
    )
    _checksums(store, prefix)
    return store.resolve(prefix)