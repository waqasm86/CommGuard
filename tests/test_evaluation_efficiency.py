from __future__ import annotations

import json

from commguard.evaluation.baselines import _adversarial_efficiency


def _manifest(root, run_id: str, family: str, designation: str, seconds: int) -> None:
    path = root / "runs" / run_id
    path.mkdir(parents=True)
    (path / "manifest.json").write_text(
        json.dumps(
            {
                "exit_status": "completed",
                "started_at_utc": "2026-08-02T00:00:00+00:00",
                "ended_at_utc": f"2026-08-02T00:00:{seconds:02d}+00:00",
                "workload_family": family,
                "designation": designation,
            }
        ),
        encoding="utf-8",
    )


def test_efficiency_uses_canonical_ddp_family_as_baseline(tmp_path) -> None:
    _manifest(tmp_path, "ddp", "ddp_training", "benign", 10)
    _manifest(tmp_path, "variant", "gradient_accumulation", "adversarial", 15)

    result = _adversarial_efficiency(tmp_path)

    assert result["baseline_family"] == "ddp_training"
    assert result["baseline_median_wall_duration_s"] == 10
    assert result["strategies"]["gradient_accumulation"]["duration_ratio_vs_baseline"] == 1.5


def test_efficiency_retains_legacy_baseline_fallback(tmp_path) -> None:
    _manifest(tmp_path, "ddp", "ddp_full_parameter", "benign", 10)
    _manifest(tmp_path, "variant", "idle_padding", "adversarial", 20)

    result = _adversarial_efficiency(tmp_path)

    assert result["baseline_family"] == "ddp_full_parameter"
    assert result["strategies"]["idle_padding"]["duration_ratio_vs_baseline"] == 2
