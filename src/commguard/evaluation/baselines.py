"""Leakage-resistant grouped baselines, ablations, metrics, and abstention."""

from __future__ import annotations

import json
import random
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from commguard.artifacts import ArtifactStore
from commguard.exceptions import CalibrationError
from commguard.features import METADATA_COLUMNS, numeric_feature_columns
from commguard.schemas import SCHEMA_VERSION, load_artifact


def grouped_split(
    rows: Iterable[dict[str, Any]],
    seed: int = 20260730,
    test_fraction: float = 0.2,
    validation_fraction: float = 0.2,
) -> dict[str, str]:
    """Assign each whole run to one split, preferring a session holdout when possible."""
    records = list(rows)
    if not records:
        raise ValueError("cannot split an empty feature collection")
    if not 0 < test_fraction < 1 or not 0 <= validation_fraction < 1:
        raise ValueError("split fractions must satisfy 0 < test < 1 and 0 <= validation < 1")
    if test_fraction + validation_fraction >= 1:
        raise ValueError("test and validation fractions must sum to less than 1")

    run_sessions: dict[str, str] = {}
    labels_by_run: dict[str, str] = {}
    for record in records:
        run_id = str(record["run_id"])
        session = str(record["session_fingerprint"])
        label = str(record["target_label"])
        if run_id in run_sessions and run_sessions[run_id] != session:
            raise ValueError(f"run_id {run_id!r} spans multiple session fingerprints")
        if run_id in labels_by_run and labels_by_run[run_id] != label:
            raise ValueError(f"run_id {run_id!r} spans multiple target labels")
        run_sessions[run_id] = session
        labels_by_run[run_id] = label

    sessions = sorted(set(run_sessions.values()))
    assignments: dict[str, str] = {}
    if len(sessions) >= 2:
        test_session = sessions[-1]
        validation_session = sessions[-2] if len(sessions) >= 3 else None
        non_test_runs = sorted(
            run_id for run_id, session in run_sessions.items() if session != test_session
        )
        random.Random(seed).shuffle(non_test_runs)
        validation_runs = (
            set()
            if validation_session is not None
            else set(non_test_runs[: max(1, round(len(non_test_runs) * validation_fraction))])
        )
        for run_id, session in run_sessions.items():
            assignments[run_id] = (
                "test"
                if session == test_session
                else "validation"
                if session == validation_session or run_id in validation_runs
                else "train"
            )
        return assignments
    by_label: dict[str, list[str]] = defaultdict(list)
    for run_id, label in labels_by_run.items():
        by_label[label].append(run_id)
    for label, run_ids in sorted(by_label.items()):
        random.Random(f"{seed}:{label}").shuffle(run_ids)
        test_count = max(1, round(len(run_ids) * test_fraction))
        validation_count = max(1, round(len(run_ids) * validation_fraction))
        if len(run_ids) < 5:
            validation_count = 0
            test_count = max(1, len(run_ids) // 3)
        for index, run_id in enumerate(run_ids):
            assignments[run_id] = (
                "test"
                if index < test_count
                else "validation"
                if index < test_count + validation_count
                else "train"
            )
    return assignments


def audit_leakage(
    rows: Iterable[dict[str, Any]],
    feature_columns: Iterable[str],
    assignments: dict[str, str],
) -> dict[str, Any]:
    records = list(rows)
    columns = list(feature_columns)
    forbidden_tokens = ("run_id", "timestamp", "path", "filename", "label", "family", "session")
    forbidden = [
        column for column in columns if any(token in column.lower() for token in forbidden_tokens)
    ]
    grouped: dict[str, set[str]] = defaultdict(set)
    for record in records:
        grouped[str(record["run_id"])].add(assignments[str(record["run_id"])])
    split_leaks = {run_id: sorted(splits) for run_id, splits in grouped.items() if len(splits) != 1}
    durations_by_label: dict[str, set[float]] = defaultdict(set)
    for record in records:
        durations_by_label[str(record["target_label"])].add(float(record["window_seconds"]))
    duration_exclusive = len({tuple(sorted(values)) for values in durations_by_label.values()}) > 1
    passed = not forbidden and not split_leaks and not duration_exclusive
    return {
        "passed": passed,
        "forbidden_feature_columns": forbidden,
        "run_split_leaks": split_leaks,
        "class_specific_window_lengths": duration_exclusive,
        "excluded_metadata_columns": sorted(METADATA_COLUMNS),
    }


def _metrics(y_true: Any, y_pred: Any) -> dict[str, Any]:
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )

    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "false_positive_rate": fp / (fp + tn) if fp + tn else None,
        "false_negative_rate": fn / (fn + tp) if fn + tp else None,
        "confusion_matrix_labels_0_nontraining_1_training": [[tn, fp], [fn, tp]],
        "sample_count": len(y_true),
    }


def _bootstrap_run_balanced_accuracy(
    run_ids: list[str],
    targets: dict[str, int],
    predictions: dict[str, int],
    repetitions: int = 500,
) -> dict[str, Any]:
    rng = random.Random(20260730)
    scores: list[float] = []
    for _ in range(repetitions):
        sampled = [rng.choice(run_ids) for _ in run_ids]
        positives = [run_id for run_id in sampled if targets[run_id] == 1]
        negatives = [run_id for run_id in sampled if targets[run_id] == 0]
        if not positives or not negatives:
            continue
        sensitivity = sum(predictions[run_id] == 1 for run_id in positives) / len(positives)
        specificity = sum(predictions[run_id] == 0 for run_id in negatives) / len(negatives)
        scores.append((sensitivity + specificity) / 2)
    scores.sort()
    if not scores:
        return {"repetitions": repetitions, "usable_repetitions": 0, "ci95": None}
    lower = scores[max(0, int(len(scores) * 0.025) - 1)]
    upper = scores[min(len(scores) - 1, int(len(scores) * 0.975))]
    return {
        "repetitions": repetitions,
        "usable_repetitions": len(scores),
        "ci95": [lower, upper],
        "median": statistics.median(scores),
        "resampling_unit": "whole run",
    }


def _simple_rule(train: Any, test: Any, columns: list[str]) -> tuple[Any, Any, dict[str, Any]]:
    import numpy as np

    candidates = [
        column
        for column in columns
        if "pcie_total_mean" in column or (
            "pcie_" in column and column.endswith("__mean")
        )
    ]
    if not candidates:
        candidates = columns[:1]
    signal = train[candidates].mean(axis=1, skipna=True)
    test_signal = test[candidates].mean(axis=1, skipna=True)
    training_median = float(signal[train["_target"] == 1].median())
    nontraining_median = float(signal[train["_target"] == 0].median())
    if not all(
        value == value for value in (training_median, nontraining_median)
    ):
        raise ValueError("simple-rule PCIe signal is entirely missing in a training class")
    threshold = (training_median + nontraining_median) / 2
    direction = 1 if training_median >= nontraining_median else -1
    finite = np.nan_to_num(test_signal.to_numpy(), nan=threshold)
    predictions = ((finite - threshold) * direction >= 0).astype(int)
    scale = max(float(np.nanstd(finite)), 1.0)
    probabilities = 1 / (1 + np.exp(-direction * (finite - threshold) / scale))
    return predictions, probabilities, {
        "columns": candidates,
        "threshold": threshold,
        "direction": "higher_is_training" if direction == 1 else "lower_is_training",
    }


def _evaluate_ablation(
    frame: Any,
    assignments: dict[str, str],
    columns: list[str],
) -> dict[str, Any]:
    import numpy as np
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    split = frame["run_id"].map(assignments)
    train = frame.loc[split == "train"].copy()
    test = frame.loc[split == "test"].copy()
    if train.empty or test.empty:
        raise ValueError("grouped split produced an empty train or test set")
    if train["_target"].nunique() < 2 or test["_target"].nunique() < 2:
        raise ValueError("train and test splits must each contain both target classes")
    x_train, y_train = train[columns], train["_target"]
    x_test, y_test = test[columns], test["_target"]
    output: dict[str, Any] = {}
    models = {
        "majority": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=2000, random_state=20260730),
        ),
        "random_forest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=2,
                max_features="sqrt",
                class_weight="balanced",
                random_state=20260730,
                n_jobs=-1,
            ),
        ),
    }
    probability_by_model: dict[str, Any] = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        prediction = model.predict(x_test)
        probability = (
            model.predict_proba(x_test)[:, list(model.classes_).index(1)]
            if hasattr(model, "predict_proba")
            else prediction.astype(float)
        )
        output[name] = _metrics(y_test.to_numpy(), prediction)
        probability_by_model[name] = probability
    simple_prediction, simple_probability, rule = _simple_rule(train, test, columns)
    output["simple_pcie_rule"] = {
        **_metrics(y_test.to_numpy(), simple_prediction),
        "rule": rule,
    }
    probability_by_model["simple_pcie_rule"] = simple_probability
    best_name = max(output, key=lambda name: output[name]["balanced_accuracy"])
    best_probability = probability_by_model[best_name]
    output["best_model"] = best_name
    output["calibration"] = {
        "brier_score": float(np.mean((best_probability - y_test.to_numpy()) ** 2))
    }
    lower, upper = 0.4, 0.6
    decided = (best_probability < lower) | (best_probability > upper)
    abstained_prediction = (best_probability[decided] >= upper).astype(int)
    output["abstention"] = {
        "lower_threshold": lower,
        "upper_threshold": upper,
        "coverage": float(decided.mean()),
        "metrics_on_decided": (
            _metrics(y_test.to_numpy()[decided], abstained_prediction)
            if decided.any()
            else None
        ),
    }
    per_family: dict[str, Any] = {}
    best_prediction = (best_probability >= 0.5).astype(int)
    for family in sorted(test["workload_family"].unique()):
        mask = test["workload_family"].to_numpy() == family
        per_family[str(family)] = _metrics(y_test.to_numpy()[mask], best_prediction[mask])
    output["per_family"] = per_family
    run_probabilities: dict[str, list[float]] = defaultdict(list)
    run_targets: dict[str, int] = {}
    for run_id, probability, target in zip(
        test["run_id"], best_probability, y_test, strict=True
    ):
        run_probabilities[str(run_id)].append(float(probability))
        run_targets[str(run_id)] = int(target)
    run_ids = sorted(run_probabilities)
    run_prediction = [
        int(sum(run_probabilities[run_id]) / len(run_probabilities[run_id]) >= 0.5)
        for run_id in run_ids
    ]
    prediction_by_run = dict(zip(run_ids, run_prediction, strict=True))
    output["run_level"] = _metrics(
        [run_targets[run_id] for run_id in run_ids], run_prediction
    )
    output["run_level"]["balanced_accuracy_bootstrap"] = _bootstrap_run_balanced_accuracy(
        run_ids,
        run_targets,
        prediction_by_run,
    )
    return output


def _heldout_adversarial_families(frame: Any, columns: list[str]) -> dict[str, Any]:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline

    output: dict[str, Any] = {}
    families = sorted(
        frame.loc[frame["designation"] == "adversarial", "workload_family"].unique()
    )
    for family in families:
        test = frame.loc[frame["workload_family"] == family]
        train = frame.loc[frame["workload_family"] != family]
        if test.empty or train["_target"].nunique() < 2:
            output[str(family)] = {"status": "insufficient_data"}
            continue
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=2,
                max_features="sqrt",
                class_weight="balanced",
                random_state=20260730,
                n_jobs=-1,
            ),
        )
        model.fit(train[columns], train["_target"])
        prediction = model.predict(test[columns])
        target = test["_target"].to_numpy()
        output[str(family)] = {
            "status": "evaluated",
            "window_count": len(test),
            "run_count": int(test["run_id"].nunique()),
            "training_detection_rate": float((prediction == 1).mean()),
            "false_negative_rate": float((prediction == 0).mean()),
            "target_is_training": bool((target == 1).all()),
            "training_excludes_complete_family": True,
        }
    return output


def _adversarial_efficiency(root: Path) -> dict[str, Any]:
    durations: dict[str, list[float]] = defaultdict(list)
    designations: dict[str, str] = {}
    for path in sorted((root / "runs").glob("*/manifest.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest["exit_status"] != "completed":
            continue
        started = datetime.fromisoformat(manifest["started_at_utc"])
        ended = datetime.fromisoformat(manifest["ended_at_utc"])
        family = str(manifest["workload_family"])
        durations[family].append((ended - started).total_seconds())
        designations[family] = str(manifest["designation"])
    baseline = durations.get("ddp_full_parameter", [])
    baseline_median = statistics.median(baseline) if baseline else None
    strategies = {}
    for family, values in sorted(durations.items()):
        if designations.get(family) != "adversarial":
            continue
        median = statistics.median(values)
        strategies[family] = {
            "completed_run_count": len(values),
            "median_wall_duration_s": median,
            "duration_ratio_vs_ddp_full_parameter": (
                median / baseline_median if baseline_median else None
            ),
        }
    return {
        "baseline_family": "ddp_full_parameter",
        "baseline_median_wall_duration_s": baseline_median,
        "strategies": strategies,
        "note": "Wall-duration ratio is a coarse measured efficiency cost, not FLOP efficiency.",
    }


def evaluate_detector(
    input_root: str | Path = "artifacts",
    output: str | Path | None = None,
    negative_calibration_mode: bool = False,
) -> dict[str, Any]:
    """Fit transparent baselines on saved feature rows using whole-run splits."""
    root = Path(input_root)
    calibration_paths = sorted((root / "results").glob("calibration-*.json"))
    calibration = (
        json.loads(calibration_paths[-1].read_text(encoding="utf-8"))
        if calibration_paths
        else None
    )
    if (
        not negative_calibration_mode
        and (calibration is None or calibration.get("status") != "supported")
    ):
        reason = "missing" if calibration is None else str(calibration.get("status"))
        raise CalibrationError(
            f"detector fitting blocked because calibration is {reason}; "
            "use explicit negative-calibration mode only for negative-result analysis"
        )
    import pandas as pd

    feature_paths = sorted((root / "features").glob("features-*.jsonl"))
    if not feature_paths:
        raise FileNotFoundError("no feature artifact found")
    loaded = load_artifact(feature_paths[-1])
    assert isinstance(loaded, list)
    rows = loaded
    frame = pd.DataFrame(rows)
    frame["_target"] = (frame["target_label"] == "training").astype(int)
    columns = numeric_feature_columns(rows)
    assignments = grouped_split(rows)
    leakage = audit_leakage(rows, columns, assignments)
    if not leakage["passed"]:
        raise ValueError(f"leakage audit failed: {leakage}")
    ablations = {
        "pcie_only": [column for column in columns if "pcie_" in column],
        "non_pcie_nvml": [column for column in columns if "pcie_" not in column],
        "combined": columns,
    }
    empty = [name for name, selected in ablations.items() if not selected]
    if empty:
        raise ValueError(f"empty feature ablations: {empty}")
    results = {
        name: _evaluate_ablation(frame, assignments, selected)
        for name, selected in ablations.items()
    }
    by_window_seconds: dict[str, Any] = {}
    for window_seconds in sorted(frame["window_seconds"].unique()):
        window_frame = frame.loc[frame["window_seconds"] == window_seconds]
        by_window_seconds[str(float(window_seconds))] = {
            name: _evaluate_ablation(window_frame, assignments, selected)
            for name, selected in ablations.items()
        }
    split_counts = Counter(assignments.values())
    result: dict[str, Any] = {
        "artifact_kind": "evaluation_result",
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "unit_of_split": "whole run; complete test-session holdout when two sessions exist",
        "calibration_status": calibration.get("status") if calibration else "missing",
        "negative_calibration_mode": negative_calibration_mode,
        "feature_source": str(feature_paths[-1]),
        "window_count": len(rows),
        "run_count": len(assignments),
        "split_run_counts": dict(split_counts),
        "split_assignments": assignments,
        "leakage_audit": leakage,
        "ablations": results,
        "by_window_seconds": by_window_seconds,
        "heldout_adversarial_families": _heldout_adversarial_families(
            frame, ablations["combined"]
        ),
        "adversarial_efficiency_cost": _adversarial_efficiency(root),
        "limitations": [
            "Scores apply only to saved workload families and recorded sessions.",
            "Window metrics are correlated within runs; run-level metrics are reported separately.",
            "No result establishes privacy, production readiness, or adversarial robustness.",
        ],
    }
    if output is not None:
        store = ArtifactStore(output)
        store.initialize()
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        store.write_json(f"results/evaluation-{stamp}.json", result)
        split_rows = [
            {
                "artifact_kind": "split_assignment",
                "schema_version": SCHEMA_VERSION,
                "run_id": run_id,
                "split": split,
            }
            for run_id, split in sorted(assignments.items())
        ]
        store.write_jsonl(f"splits/assignments-{stamp}.jsonl", split_rows)
    return result
