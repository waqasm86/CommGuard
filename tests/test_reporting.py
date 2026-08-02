from __future__ import annotations

import json

from commguard.reporting import generate_report


def test_report_uses_primary_communication_result_and_saved_coverage(tmp_path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    (results / "matrix-standard-test.json").write_text(
        json.dumps(
            {
                "family_counts": {
                    "ddp_training": {
                        "planned": 3,
                        "completed": 3,
                        "failed": 0,
                        "feature_valid": 3,
                        "feature_excluded": 0,
                    }
                },
                "primary_coverage_gate": {"passed": True},
            }
        ),
        encoding="utf-8",
    )
    primary = {
        "selected_model": "logistic_regression",
        "window_level": {
            "balanced_accuracy": 0.75,
            "false_positive_rate": 0.25,
            "false_negative_rate": 0.25,
        },
        "run_level": {
            "balanced_accuracy": 0.5,
            "false_positive_rate": 0.0,
            "false_negative_rate": 1.0,
        },
        "per_family": {
            "ddp_training": {
                "sample_count": 1,
                "balanced_accuracy": None,
                "false_positive_rate": None,
                "false_negative_rate": 1.0,
            }
        },
    }
    (results / "evaluation-test.json").write_text(
        json.dumps(
            {
                "primary_communication_only": {"metrics": primary},
                "ablations": {"primary_communication_only": primary},
                "split_plan": {
                    "actual_strategy": "independent_session_holdout",
                    "run_ids_by_split": {
                        "train": ["run-train"],
                        "validation": ["run-validation"],
                        "test": ["run-test"],
                    },
                    "session_ids_by_split": {
                        "train": ["session-train"],
                        "validation": ["session-validation"],
                        "test": ["session-test"],
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    report = generate_report(tmp_path)

    assert "primary `communication_only` result selected `logistic_regression`" in report
    assert "run false-negative rate: 1.000" in report
    assert "| `ddp_training` | 3 | 3 | 0 | 3 | 0 |" in report
    assert "Actual split strategy: `independent_session_holdout`" in report
    assert "| `ddp_training` | 1 | unavailable | unavailable | 1.000 |" in report
    assert "`results/evaluation-test.json`: `" in report


def test_report_retains_legacy_pcie_primary_boundary(tmp_path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    legacy = {
        "best_model": "majority",
        "majority": {"balanced_accuracy": 0.5},
        "run_level": {
            "balanced_accuracy": 0.5,
            "false_positive_rate": 0.0,
            "false_negative_rate": 1.0,
        },
    }
    (results / "evaluation-legacy.json").write_text(
        json.dumps({"ablations": {"pcie_only": legacy, "combined": legacy}}),
        encoding="utf-8",
    )

    report = generate_report(tmp_path)

    assert "primary `pcie_only_legacy` result" in report
    assert "combined-signal best baseline" not in report
