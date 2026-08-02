"""Leakage-resistant grouped baselines, ablations, metrics, and abstention."""

from __future__ import annotations

import json
import random
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from commguard.artifacts import ArtifactStore
from commguard.evaluation.splits import audit_leakage, make_split_plan
from commguard.exceptions import CalibrationError
from commguard.features import (
    PRIMARY_BENIGN_FAMILIES,
    load_extraction_result,
    numeric_feature_columns,
    require_primary_coverage,
)
from commguard.schemas import CURRENT_SCHEMA_VERSION, SCHEMA_VERSION


def _metrics(y_true: Any, y_pred: Any, probabilities: Any | None = None) -> dict[str, Any]:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    classes = set(int(value) for value in y_true)
    has_both_classes = classes == {0, 1}
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": (
            float(balanced_accuracy_score(y_true, y_pred)) if has_both_classes else None
        ),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "false_positive_rate": fp / (fp + tn) if fp + tn else None,
        "false_negative_rate": fn / (fn + tp) if fn + tp else None,
        "class_recall": {
            "nontraining": tn / (tn + fp) if tn + fp else None,
            "training": tp / (tp + fn) if tp + fn else None,
        },
        "class_precision": {
            "nontraining": tn / (tn + fn) if tn + fn else None,
            "training": tp / (tp + fp) if tp + fp else None,
        },
        "confusion_matrix_labels_0_nontraining_1_training": [[tn, fp], [fn, tp]],
        "sample_count": len(y_true),
        "class_counts": {"nontraining": tn + fp, "training": fn + tp},
        "auroc": (
            float(roc_auc_score(y_true, probabilities))
            if has_both_classes and probabilities is not None
            else None
        ),
        "auprc": (
            float(average_precision_score(y_true, probabilities))
            if has_both_classes and probabilities is not None
            else None
        ),
        "warning": None if has_both_classes else "one target class; balanced metrics undefined",
    }


def _bootstrap_run_balanced_accuracy(
    run_ids: list[str],
    targets: dict[str, int],
    predictions: dict[str, int],
    repetitions: int = 500,
    minimum_independent_groups: int = 20,
) -> dict[str, Any]:
    if len(run_ids) < minimum_independent_groups:
        return {
            "repetitions": 0,
            "usable_repetitions": 0,
            "ci95": None,
            "resampling_unit": "whole run",
            "independent_group_count": len(run_ids),
            "warning": (
                "insufficient independent groups for an uncertainty interval: "
                f"observed={len(run_ids)} required={minimum_independent_groups}"
            ),
        }
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
        "independent_group_count": len(run_ids),
        "warning": None,
    }


def _validation_balanced_accuracy(
    targets: list[int], probabilities: list[float], threshold: float
) -> float:
    positives = [index for index, target in enumerate(targets) if target == 1]
    negatives = [index for index, target in enumerate(targets) if target == 0]
    if not positives or not negatives:
        raise ValueError("validation selection requires both target classes")
    sensitivity = sum(probabilities[index] >= threshold for index in positives) / len(positives)
    specificity = sum(probabilities[index] < threshold for index in negatives) / len(negatives)
    return (sensitivity + specificity) / 2


def _select_model_and_threshold(
    validation_targets: list[int],
    validation_probabilities: dict[str, list[float]],
) -> tuple[str, float, dict[str, float]]:
    """Select solely from validation inputs; final-test values are not accepted."""
    if not validation_probabilities:
        raise ValueError("no validation model candidates were supplied")
    model_scores = {
        name: _validation_balanced_accuracy(validation_targets, values, 0.5)
        for name, values in validation_probabilities.items()
    }
    order = list(validation_probabilities)
    selected_model = max(order, key=lambda name: (model_scores[name], -order.index(name)))
    threshold_candidates = [value / 20 for value in range(2, 19)]
    selected_threshold = max(
        threshold_candidates,
        key=lambda threshold: (
            _validation_balanced_accuracy(
                validation_targets,
                validation_probabilities[selected_model],
                threshold,
            ),
            -abs(threshold - 0.5),
        ),
    )
    return selected_model, selected_threshold, model_scores


def _simple_rule(
    train: Any, evaluation: Any, columns: list[str]
) -> tuple[Any, Any, dict[str, Any]]:
    import numpy as np

    candidates = [
        column
        for column in columns
        if "pcie_total_mean" in column or ("pcie_" in column and column.endswith("__mean"))
    ]
    if not candidates:
        candidates = columns[:1]
    signal = train[candidates].mean(axis=1, skipna=True)
    evaluation_signal = evaluation[candidates].mean(axis=1, skipna=True)
    training_median = float(signal[train["_target"] == 1].median())
    nontraining_median = float(signal[train["_target"] == 0].median())
    if not all(value == value for value in (training_median, nontraining_median)):
        raise ValueError("simple-rule PCIe signal is entirely missing in a training class")
    threshold = (training_median + nontraining_median) / 2
    direction = 1 if training_median >= nontraining_median else -1
    finite = np.nan_to_num(evaluation_signal.to_numpy(), nan=threshold)
    predictions = ((finite - threshold) * direction >= 0).astype(int)
    train_finite = np.nan_to_num(signal.to_numpy(), nan=threshold)
    scale = max(float(np.nanstd(train_finite)), 1.0)
    probabilities = 1 / (1 + np.exp(-direction * (finite - threshold) / scale))
    return (
        predictions,
        probabilities,
        {
            "columns": candidates,
            "threshold": threshold,
            "direction": "higher_is_training" if direction == 1 else "lower_is_training",
        },
    )


def _evaluate_ablation(
    frame: Any,
    assignments: dict[str, str],
    columns: list[str],
    *,
    diagnostic_only: bool = False,
) -> dict[str, Any]:
    import numpy as np
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    split = frame["run_id"].map(assignments)
    train = frame.loc[split == "train"].copy()
    validation = frame.loc[split == "validation"].copy()
    test = frame.loc[split == "test"].copy()
    if train.empty or validation.empty or test.empty:
        raise ValueError("grouped split produced an empty train, validation, or test set")
    if train["_target"].nunique() < 2 or validation["_target"].nunique() < 2:
        raise ValueError("train and validation splits must each contain both target classes")
    if not diagnostic_only and test["_target"].nunique() < 2:
        raise ValueError("primary test split must contain both target classes")
    selected_columns = [column for column in columns if train[column].notna().any()]
    dropped_all_missing = sorted(set(columns) - set(selected_columns))
    if not selected_columns:
        raise ValueError("all candidate features are missing in the training partition")
    x_train, y_train = train[columns], train["_target"]
    x_train = train[selected_columns]
    y_validation = validation["_target"]
    y_test = test["_target"]
    models = {
        "majority": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": make_pipeline(
            SimpleImputer(strategy="median"),
            VarianceThreshold(),
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=2000, random_state=20260730),
        ),
        "random_forest": make_pipeline(
            SimpleImputer(strategy="median"),
            VarianceThreshold(),
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
    validation_probability_by_model: dict[str, Any] = {}
    test_probability_by_model: dict[str, Any] = {}
    candidate_results: dict[str, Any] = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        validation_prediction = model.predict(validation[selected_columns])
        validation_probability = (
            model.predict_proba(validation[selected_columns])[:, list(model.classes_).index(1)]
            if hasattr(model, "predict_proba")
            else validation_prediction.astype(float)
        )
        test_prediction = model.predict(test[selected_columns])
        test_probability = (
            model.predict_proba(test[selected_columns])[:, list(model.classes_).index(1)]
            if hasattr(model, "predict_proba")
            else test_prediction.astype(float)
        )
        candidate_results[name] = {
            "validation": _metrics(
                y_validation.to_numpy(), validation_prediction, validation_probability
            ),
            "test": _metrics(y_test.to_numpy(), test_prediction, test_probability),
        }
        validation_probability_by_model[name] = validation_probability
        test_probability_by_model[name] = test_probability
    validation_simple_prediction, validation_simple_probability, rule = _simple_rule(
        train, validation, selected_columns
    )
    test_simple_prediction, test_simple_probability, _ = _simple_rule(train, test, selected_columns)
    candidate_results["simple_rule"] = {
        "validation": _metrics(
            y_validation.to_numpy(),
            validation_simple_prediction,
            validation_simple_probability,
        ),
        "test": _metrics(y_test.to_numpy(), test_simple_prediction, test_simple_probability),
        "rule_fitted_on_train": rule,
    }
    validation_probability_by_model["simple_rule"] = validation_simple_probability
    test_probability_by_model["simple_rule"] = test_simple_probability
    best_name, selected_threshold, validation_selection_scores = _select_model_and_threshold(
        [int(value) for value in y_validation],
        {
            name: [float(value) for value in probabilities]
            for name, probabilities in validation_probability_by_model.items()
        },
    )
    validation_probability = validation_probability_by_model[best_name]
    test_probability = test_probability_by_model[best_name]
    best_prediction = (test_probability >= selected_threshold).astype(int)
    output: dict[str, Any] = {
        "selection_protocol": {
            "model_selected_on": "validation",
            "threshold_selected_on": "validation",
            "test_used_for_selection": False,
            "probability_calibration": "none",
        },
        "feature_preprocessing": {
            "fit_partition": "train",
            "input_feature_count": len(columns),
            "usable_training_feature_count": len(selected_columns),
            "dropped_all_missing_training_features": dropped_all_missing,
            "imputation": "training median",
            "scaling": "training fit for logistic regression",
            "variance_filter": "training fit",
        },
        "candidate_models": candidate_results,
        "selected_model": best_name,
        "selected_probability_threshold": selected_threshold,
        "validation_selection_balanced_accuracy": validation_selection_scores,
        "window_level": _metrics(y_test.to_numpy(), best_prediction, test_probability),
        "probability_quality_diagnostic": {
            "test_brier_score": float(np.mean((test_probability - y_test.to_numpy()) ** 2)),
            "probabilities_are_calibrated": False,
        },
        "sample_counts": {
            "train_windows": len(train),
            "validation_windows": len(validation),
            "test_windows": len(test),
            "train_runs": int(train["run_id"].nunique()),
            "validation_runs": int(validation["run_id"].nunique()),
            "test_runs": int(test["run_id"].nunique()),
        },
    }
    lower = max(0.0, selected_threshold - 0.1)
    upper = min(1.0, selected_threshold + 0.1)
    decided = (test_probability < lower) | (test_probability > upper)
    abstained_prediction = (test_probability[decided] >= selected_threshold).astype(int)
    output["abstention"] = {
        "lower_threshold": lower,
        "upper_threshold": upper,
        "coverage": float(decided.mean()),
        "selective_risk": (
            float((abstained_prediction != y_test.to_numpy()[decided]).mean())
            if decided.any()
            else None
        ),
        "metrics_on_decided": (
            _metrics(y_test.to_numpy()[decided], abstained_prediction) if decided.any() else None
        ),
    }
    per_family: dict[str, Any] = {}
    for family in sorted(test["workload_family"].unique()):
        mask = test["workload_family"].to_numpy() == family
        per_family[str(family)] = _metrics(
            y_test.to_numpy()[mask],
            best_prediction[mask],
            test_probability[mask],
        )
    output["per_family"] = per_family
    run_probabilities: dict[str, list[float]] = defaultdict(list)
    run_targets: dict[str, int] = {}
    for run_id, probability, target in zip(test["run_id"], test_probability, y_test, strict=True):
        run_probabilities[str(run_id)].append(float(probability))
        run_targets[str(run_id)] = int(target)
    run_ids = sorted(run_probabilities)
    run_prediction = [
        int(sum(run_probabilities[run_id]) / len(run_probabilities[run_id]) >= selected_threshold)
        for run_id in run_ids
    ]
    prediction_by_run = dict(zip(run_ids, run_prediction, strict=True))
    run_probability = [
        sum(run_probabilities[run_id]) / len(run_probabilities[run_id]) for run_id in run_ids
    ]
    output["run_level"] = _metrics(
        [run_targets[run_id] for run_id in run_ids],
        run_prediction,
        run_probability,
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
    families = sorted(frame.loc[frame["designation"] == "adversarial", "workload_family"].unique())
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


def _hard_negative_false_positive_rates(per_family: dict[str, Any]) -> dict[str, Any]:
    aliases = {
        "host_transfer": ("control_host_transfer", "host_device_transfer"),
        "synchronized_inference": ("inference_synchronized", "synchronized_prefill"),
    }
    result: dict[str, Any] = {}
    for name, families in aliases.items():
        matches = [per_family[family] for family in families if family in per_family]
        if not matches:
            result[name] = {
                "false_positive_rate": None,
                "sample_count": 0,
                "warning": "family absent from the test partition",
            }
            continue
        false_positives = sum(
            metrics["confusion_matrix_labels_0_nontraining_1_training"][0][1] for metrics in matches
        )
        negatives = sum(metrics["class_counts"]["nontraining"] for metrics in matches)
        result[name] = {
            "false_positive_rate": false_positives / negatives if negatives else None,
            "sample_count": negatives,
            "warning": None if negatives else "family has no non-training test samples",
        }
    return result


def _robustness_holdouts(
    frame: Any,
    columns: list[str],
    *,
    mode: str,
) -> dict[str, Any]:
    identity = "workload_family" if mode == "family_holdout" else "workload_config_id"
    output: dict[str, Any] = {}
    for value in sorted(str(item) for item in frame[identity].dropna().unique()):
        try:
            plan = make_split_plan(
                frame.to_dict("records"),
                mode=mode,
                holdout_value=value,
            )
            metrics = _evaluate_ablation(
                frame,
                plan.assignments,
                columns,
                diagnostic_only=True,
            )
            output[value] = {
                "status": "evaluated_diagnostic",
                "split_plan": plan.to_dict(),
                "metrics": metrics,
            }
        except ValueError as exc:
            output[value] = {
                "status": "insufficient_data",
                "reason": str(exc),
            }
    return output


def evaluate_detector(
    input_root: str | Path = "artifacts",
    output: str | Path | None = None,
    negative_calibration_mode: bool = False,
    required_families: tuple[str, ...] = PRIMARY_BENIGN_FAMILIES,
    minimum_runs_per_family: int = 3,
) -> dict[str, Any]:
    """Fit transparent baselines on saved feature rows using whole-run splits."""
    root = Path(input_root)
    calibration_paths = sorted((root / "results").glob("calibration-*.json"))
    calibration = (
        json.loads(calibration_paths[-1].read_text(encoding="utf-8")) if calibration_paths else None
    )
    if not negative_calibration_mode and (
        calibration is None or calibration.get("status") != "supported"
    ):
        reason = "missing" if calibration is None else str(calibration.get("status"))
        raise CalibrationError(
            f"detector fitting blocked because calibration is {reason}; "
            "use explicit negative-calibration mode only for negative-result analysis"
        )
    extraction = load_extraction_result(root)
    coverage_gate = require_primary_coverage(
        extraction,
        required_families=required_families,
        minimum_runs_per_family=minimum_runs_per_family,
    )
    import pandas as pd

    rows = list(extraction.features)
    feature_paths = sorted((root / "features").glob("features-*.jsonl"))
    frame = pd.DataFrame(rows)
    frame["_target"] = (frame["target_label"] == "training").astype(int)
    primary_frame = frame.loc[frame["window_seconds"] == 30.0].copy()
    if primary_frame.empty:
        raise ValueError("primary 30-second feature frame is empty after coverage passed")
    primary_rows = primary_frame.to_dict("records")
    columns = numeric_feature_columns(primary_rows)
    split_plan = make_split_plan(primary_rows, required_families=required_families)
    assignments = split_plan.assignments
    leakage = audit_leakage(primary_rows, columns, assignments)
    if not leakage["passed"]:
        raise ValueError(f"leakage audit failed: {leakage}")
    ablations = {
        "communication_only": [column for column in columns if "pcie_" in column],
        "non_pcie_nvml": [column for column in columns if "pcie_" not in column],
        "combined": columns,
    }
    empty = [name for name, selected in ablations.items() if not selected]
    if empty:
        raise ValueError(f"empty feature ablations: {empty}")
    primary_communication = _evaluate_ablation(
        primary_frame,
        assignments,
        ablations["communication_only"],
    )
    diagnostic_ablations = {
        name: _evaluate_ablation(primary_frame, assignments, selected)
        for name, selected in ablations.items()
        if name != "communication_only"
    }
    by_window_seconds: dict[str, Any] = {}
    for window_seconds in sorted(frame["window_seconds"].unique()):
        window_frame = frame.loc[frame["window_seconds"] == window_seconds]
        by_window_seconds[str(float(window_seconds))] = _evaluate_ablation(
            window_frame,
            assignments,
            ablations["communication_only"],
        )
    split_counts = Counter(assignments.values())
    independent_test_runs = len(split_plan.run_ids_by_split["test"])
    small_sample_warning = (
        "insufficient independent groups for stable generalization estimates: "
        f"test_runs={independent_test_runs} recommended_minimum=20"
        if independent_test_runs < 20
        else None
    )
    hard_negatives = _hard_negative_false_positive_rates(primary_communication["per_family"])
    result: dict[str, Any] = {
        "artifact_kind": "evaluation_result",
        "schema_version": CURRENT_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "primary_communication_only": {
            "window_seconds": 30.0,
            "feature_columns": ablations["communication_only"],
            "metrics": primary_communication,
            "hard_negative_false_positive_rates": hard_negatives,
        },
        "unit_of_split": "whole run",
        "actual_split_strategy": split_plan.actual_strategy,
        "split_plan": split_plan.to_dict(),
        "calibration_status": calibration.get("status") if calibration else "missing",
        "negative_calibration_mode": negative_calibration_mode,
        "feature_source": str(feature_paths[-1]),
        "coverage_gate": coverage_gate,
        "window_count": len(rows),
        "run_count": len(assignments),
        "split_run_counts": dict(split_counts),
        "split_assignments": assignments,
        "leakage_audit": leakage,
        "ablations": {
            "primary_communication_only": primary_communication,
            **diagnostic_ablations,
        },
        "diagnostic_by_window_seconds_communication_only": by_window_seconds,
        "robustness_evaluations": {
            "session_holdout": {
                "status": (
                    "primary_strategy"
                    if split_plan.actual_strategy == "independent_session_holdout"
                    else "unavailable"
                ),
                "reason": (
                    None
                    if split_plan.actual_strategy == "independent_session_holdout"
                    else "fewer than three valid sessions or class/family coverage was insufficient"
                ),
            },
            "family_holdout": _robustness_holdouts(
                primary_frame,
                ablations["communication_only"],
                mode="family_holdout",
            ),
            "configuration_holdout": _robustness_holdouts(
                primary_frame,
                ablations["communication_only"],
                mode="configuration_holdout",
            ),
        },
        "heldout_adversarial_families": _heldout_adversarial_families(frame, ablations["combined"]),
        "adversarial_efficiency_cost": _adversarial_efficiency(root),
        "warnings": [
            warning
            for warning in [small_sample_warning, *split_plan.warnings]
            if warning is not None
        ],
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
                "actual_strategy": split_plan.actual_strategy,
            }
            for run_id, split in sorted(assignments.items())
        ]
        store.write_jsonl(f"splits/assignments-{stamp}.jsonl", split_rows)
    return result
