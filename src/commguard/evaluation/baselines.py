"""Leakage-resistant grouped baselines, ablations, metrics, and abstention."""

from __future__ import annotations

import json
import random
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from commguard.adversarial import AdversarialHoldoutPlan
from commguard.artifacts import ArtifactStore
from commguard.calibration import verify_calibration_reference
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


def _stable_sigmoid(raw_logits: Any) -> Any:
    """Return finite logistic probabilities without overflow warnings."""
    import numpy as np

    logits = np.clip(np.asarray(raw_logits, dtype=float), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-logits))


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
    probabilities = _stable_sigmoid(direction * (finite - threshold) / scale)
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


def _adversarial_outcome_summary(
    target_label: str,
    probabilities: list[float],
    threshold: float,
    *,
    run_count: int,
) -> dict[str, Any]:
    if target_label not in {"training", "control", "inference"}:
        raise ValueError(f"unsupported adversarial target label {target_label!r}")
    if not probabilities:
        raise ValueError("adversarial outcome requires at least one detector score")
    predictions = [int(value >= threshold) for value in probabilities]
    target_is_training = target_label == "training"
    expected = 1 if target_is_training else 0
    error_rate = sum(prediction != expected for prediction in predictions) / len(predictions)
    return {
        "target_label": target_label,
        "target_is_training": target_is_training,
        "window_count": len(probabilities),
        "run_count": run_count,
        "detector_score_mean": statistics.fmean(probabilities),
        "detector_score_median": statistics.median(probabilities),
        "detector_threshold": threshold,
        "evasion_rate": error_rate if target_is_training else None,
        "false_negative_rate": error_rate if target_is_training else None,
        "false_positive_rate": error_rate if not target_is_training else None,
        "training_detection_rate": 1.0 - error_rate if target_is_training else None,
    }


def _partition_primary_evaluation_rows(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Separate benign fitting rows from adversarial scoring rows before pandas/model code."""
    benign = [
        row
        for row in rows
        if float(row.get("window_seconds", -1)) == 30.0 and row.get("designation") == "benign"
    ]
    adversarial = [
        row
        for row in rows
        if float(row.get("window_seconds", -1)) == 30.0 and row.get("designation") == "adversarial"
    ]
    return benign, adversarial


def _heldout_adversarial_families(
    benign_frame: Any,
    adversarial_frame: Any,
    assignments: dict[str, str],
    columns: list[str],
    holdout_plan: AdversarialHoldoutPlan | None,
    *,
    release_final_holdout: bool,
) -> dict[str, Any]:
    if adversarial_frame.empty:
        return {}
    families = sorted(str(value) for value in adversarial_frame["workload_family"].unique())
    if holdout_plan is None:
        return {
            family: {
                "status": "blocked_missing_adversarial_holdout_plan",
                "reason": "adversarial evidence is present but no sealed round plan was supplied",
            }
            for family in families
        }
    holdout_plan.validate()
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline

    split = benign_frame["run_id"].map(assignments)
    train = benign_frame.loc[split == "train"].copy()
    final_identity = train.apply(
        lambda row: holdout_plan.is_final_identity(
            family=str(row["workload_family"]),
            session_id=str(row["experiment_session_id"]),
            config_id=str(row["workload_config_id"]),
        ),
        axis=1,
    )
    train = train.loc[~final_identity].copy()
    if train.empty or train["_target"].nunique() < 2:
        raise ValueError("frozen adversarial baseline requires both benign target classes in train")
    selected_columns = [column for column in columns if train[column].notna().any()]
    if not selected_columns:
        raise ValueError("frozen adversarial baseline has no usable benign training features")
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
    model.fit(train[selected_columns], train["_target"])
    threshold = 0.5
    output: dict[str, Any] = {}
    for family in families:
        family_frame = adversarial_frame.loc[adversarial_frame["workload_family"] == family]
        round_names = {
            holdout_plan.round_for(
                family=family,
                session_id=str(row["experiment_session_id"]),
                config_id=str(row["workload_config_id"]),
                release_final=release_final_holdout,
            )
            for _, row in family_frame.iterrows()
        }
        if "sealed_final_holdout" in round_names:
            output[family] = {
                "status": "sealed_final_holdout",
                "window_count": len(family_frame),
                "run_count": int(family_frame["run_id"].nunique()),
                "detector_score": None,
                "final_data_used_for_fitting_or_selection": False,
            }
            continue
        target_labels = {str(value) for value in family_frame["target_label"].unique()}
        if len(target_labels) != 1:
            output[family] = {
                "status": "invalid_mixed_target_family",
                "target_labels": sorted(target_labels),
            }
            continue
        probabilities = model.predict_proba(family_frame[selected_columns])[
            :, list(model.classes_).index(1)
        ]
        output[family] = {
            "status": "evaluated_frozen_benign_only_baseline",
            "rounds": sorted(round_names),
            "model": "random_forest_fixed_v1",
            "fit_rows": "benign_primary_train_only",
            "final_session_or_config_rows_used_for_fitting": False,
            "adversarial_data_used_for_fitting_or_selection": False,
            **_adversarial_outcome_summary(
                next(iter(target_labels)),
                [float(value) for value in probabilities],
                threshold,
                run_count=int(family_frame["run_id"].nunique()),
            ),
        }
    return output


def _adversarial_efficiency(root: Path) -> dict[str, Any]:
    durations: dict[str, list[float]] = defaultdict(list)
    designations: dict[str, str] = {}
    strategy_metrics: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for path in sorted((root / "runs").glob("*/manifest.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest["exit_status"] != "completed":
            continue
        started = datetime.fromisoformat(manifest["started_at_utc"])
        ended = datetime.fromisoformat(manifest["ended_at_utc"])
        family = str(manifest["workload_family"])
        durations[family].append((ended - started).total_seconds())
        designations[family] = str(manifest["designation"])
        if manifest["designation"] == "adversarial":
            rank_evidence = manifest.get("config", {}).get("rank_runtime_evidence", {})
            for evidence in rank_evidence.values():
                summary = evidence.get("strategy_summary", {})
                memory = evidence.get("memory_peak", {})
                for source_name, output_name in (
                    ("actual_sync_rounds", "sync_rounds_per_rank"),
                    ("communication_bytes_proxy", "communication_bytes_proxy_per_rank"),
                    ("throughput_tokens_per_s", "throughput_tokens_per_s_per_rank"),
                    ("final_loss_proxy", "final_loss_proxy_per_rank"),
                ):
                    value = summary.get(source_name)
                    if isinstance(value, (int, float)):
                        strategy_metrics[family][output_name].append(float(value))
                allocated = memory.get("allocated_bytes")
                if isinstance(allocated, int):
                    strategy_metrics[family]["peak_allocated_bytes_per_rank"].append(
                        float(allocated)
                    )
    baseline_family = "ddp_training" if durations.get("ddp_training") else "ddp_full_parameter"
    baseline = durations.get(baseline_family, [])
    baseline_median = statistics.median(baseline) if baseline else None
    strategies = {}
    for family, values in sorted(durations.items()):
        if designations.get(family) != "adversarial":
            continue
        median = statistics.median(values)
        strategies[family] = {
            "completed_run_count": len(values),
            "median_wall_duration_s": median,
            "duration_ratio_vs_baseline": median / baseline_median if baseline_median else None,
            "measured_proxy_medians": {
                name: statistics.median(metric_values)
                for name, metric_values in sorted(strategy_metrics[family].items())
                if metric_values
            },
        }
    return {
        "baseline_family": baseline_family,
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
    adversarial_holdout_plan: AdversarialHoldoutPlan | None = None,
    release_final_adversarial_holdout: bool = False,
    benign_extraction_summary: str | Path | None = None,
    adversarial_extraction_summary: str | Path | None = None,
) -> dict[str, Any]:
    """Fit transparent baselines on saved feature rows using whole-run splits."""
    root = Path(input_root)
    extraction = load_extraction_result(root, benign_extraction_summary)
    calibration_reference = extraction.calibration_reference
    calibration: dict[str, Any] | None = None
    calibration_path: Path | None = None
    if calibration_reference is None:
        if not negative_calibration_mode:
            raise CalibrationError(
                "detector fitting blocked because the extraction calibration reference "
                "is missing; legacy evidence requires explicit negative-calibration mode"
            )
    else:
        sessions = {
            str(record.experiment_session_id)
            for record in extraction.coverage
            if record.experiment_session_id
        }
        environments = {
            str(row["environment_fingerprint"])
            for row in extraction.features
            if row.get("environment_fingerprint")
        }
        source_commits = {
            str(row["source_commit"]) for row in extraction.features if row.get("source_commit")
        }
        calibration_path, calibration = verify_calibration_reference(
            root,
            calibration_reference,
            expected_experiment_session_ids=sessions or None,
            expected_environment_fingerprints=environments or None,
            expected_source_commits=source_commits or None,
            require_current_session=not negative_calibration_mode,
            allow_legacy=negative_calibration_mode,
        )
    coverage_gate = require_primary_coverage(
        extraction,
        required_families=required_families,
        minimum_runs_per_family=minimum_runs_per_family,
    )
    import pandas as pd

    rows = list(extraction.features)
    if adversarial_extraction_summary is not None:
        adversarial_extraction = load_extraction_result(root, adversarial_extraction_summary)
        if set(adversarial_extraction.selected_designations) != {"adversarial"}:
            raise ValueError("adversarial extraction must select only the adversarial designation")
        rows.extend(adversarial_extraction.features)
    benign_primary_rows, adversarial_primary_rows = _partition_primary_evaluation_rows(rows)
    feature_paths = sorted((root / "features").glob("features-*.jsonl"))
    frame = pd.DataFrame(rows)
    frame["_target"] = (frame["target_label"] == "training").astype(int)
    primary_frame = pd.DataFrame(benign_primary_rows)
    adversarial_frame = pd.DataFrame(adversarial_primary_rows)
    if not primary_frame.empty:
        primary_frame["_target"] = (primary_frame["target_label"] == "training").astype(int)
    if not adversarial_frame.empty:
        adversarial_frame["_target"] = (adversarial_frame["target_label"] == "training").astype(int)
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
    benign_frame = frame.loc[frame["designation"] == "benign"]
    for window_seconds in sorted(benign_frame["window_seconds"].unique()):
        window_frame = benign_frame.loc[benign_frame["window_seconds"] == window_seconds]
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
        "calibration_reference": calibration_reference,
        "calibration_artifact": (
            str(calibration_path.relative_to(root)) if calibration_path is not None else None
        ),
        "negative_calibration_mode": negative_calibration_mode,
        "feature_source": (
            str(benign_extraction_summary)
            if benign_extraction_summary is not None
            else str(feature_paths[-1])
        ),
        "adversarial_feature_source": (
            str(adversarial_extraction_summary)
            if adversarial_extraction_summary is not None
            else None
        ),
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
        "adversarial_holdout_plan": (
            adversarial_holdout_plan.to_dict() if adversarial_holdout_plan is not None else None
        ),
        "final_adversarial_holdout_released": release_final_adversarial_holdout,
        "heldout_adversarial_families": _heldout_adversarial_families(
            primary_frame,
            adversarial_frame,
            assignments,
            ablations["combined"],
            adversarial_holdout_plan,
            release_final_holdout=release_final_adversarial_holdout,
        ),
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
        result_artifact = f"results/evaluation-{stamp}.json"
        result["result_artifact"] = result_artifact
        store.write_json(result_artifact, result)
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
