from __future__ import annotations

import json

from commguard.evaluation.baselines import (
    _adversarial_efficiency,
    _adversarial_outcome_summary,
    _partition_primary_evaluation_rows,
)


def _manifest(
    root,
    run_id: str,
    family: str,
    designation: str,
    seconds: int,
    *,
    with_strategy_metrics: bool = False,
) -> None:
    path = root / "runs" / run_id
    path.mkdir(parents=True)
    config = {}
    if with_strategy_metrics:
        config = {
            "rank_runtime_evidence": {
                rank: {
                    "strategy_summary": {
                        "actual_sync_rounds": 3,
                        "communication_bytes_proxy": 1024,
                        "throughput_tokens_per_s": 100.0,
                        "final_loss_proxy": 1.5,
                    },
                    "memory_peak": {"allocated_bytes": 2048},
                }
                for rank in ("0", "1")
            }
        }
    (path / "manifest.json").write_text(
        json.dumps(
            {
                "exit_status": "completed",
                "started_at_utc": "2026-08-02T00:00:00+00:00",
                "ended_at_utc": f"2026-08-02T00:00:{seconds:02d}+00:00",
                "workload_family": family,
                "designation": designation,
                "config": config,
            }
        ),
        encoding="utf-8",
    )


def test_efficiency_uses_canonical_ddp_family_as_baseline(tmp_path) -> None:
    _manifest(tmp_path, "ddp", "ddp_training", "benign", 10)
    _manifest(
        tmp_path,
        "variant",
        "gradient_accumulation",
        "adversarial",
        15,
        with_strategy_metrics=True,
    )

    result = _adversarial_efficiency(tmp_path)

    assert result["baseline_family"] == "ddp_training"
    assert result["baseline_median_wall_duration_s"] == 10
    assert result["strategies"]["gradient_accumulation"]["duration_ratio_vs_baseline"] == 1.5
    assert result["strategies"]["gradient_accumulation"]["measured_proxy_medians"] == {
        "communication_bytes_proxy_per_rank": 1024.0,
        "final_loss_proxy_per_rank": 1.5,
        "peak_allocated_bytes_per_rank": 2048.0,
        "sync_rounds_per_rank": 3.0,
        "throughput_tokens_per_s_per_rank": 100.0,
    }


def test_efficiency_retains_legacy_baseline_fallback(tmp_path) -> None:
    _manifest(tmp_path, "ddp", "ddp_full_parameter", "benign", 10)
    _manifest(tmp_path, "variant", "idle_padding", "adversarial", 20)

    result = _adversarial_efficiency(tmp_path)

    assert result["baseline_family"] == "ddp_full_parameter"
    assert result["strategies"]["idle_padding"]["duration_ratio_vs_baseline"] == 2


def test_adversarial_outcomes_distinguish_evasion_from_decoy_false_positives() -> None:
    training = _adversarial_outcome_summary("training", [0.2, 0.8], 0.5, run_count=2)
    decoy = _adversarial_outcome_summary("control", [0.9, 0.1], 0.5, run_count=2)

    assert training["evasion_rate"] == 0.5
    assert training["false_negative_rate"] == 0.5
    assert training["false_positive_rate"] is None
    assert decoy["evasion_rate"] is None
    assert decoy["false_positive_rate"] == 0.5


def test_adversarial_rows_never_enter_primary_benign_fitting_partition() -> None:
    rows = [
        {"run_id": "benign", "window_seconds": 30.0, "designation": "benign"},
        {"run_id": "adversarial", "window_seconds": 30.0, "designation": "adversarial"},
        {"run_id": "short", "window_seconds": 15.0, "designation": "benign"},
        {"run_id": "calibration", "window_seconds": 30.0, "designation": "calibration"},
    ]

    benign, adversarial = _partition_primary_evaluation_rows(rows)

    assert [row["run_id"] for row in benign] == ["benign"]
    assert [row["run_id"] for row in adversarial] == ["adversarial"]
