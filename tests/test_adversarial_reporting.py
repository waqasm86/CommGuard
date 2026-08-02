from __future__ import annotations

from commguard.reporting.report import _adversarial_result_text


def test_adversarial_report_distinguishes_evasion_decoy_and_sealed_results() -> None:
    assert (
        _adversarial_result_text(
            "periodic_local_sgd",
            {
                "status": "evaluated_frozen_benign_only_baseline",
                "target_is_training": True,
                "evasion_rate": 0.25,
            },
        )
        == "periodic_local_sgd: evasion rate 0.25"
    )
    assert (
        _adversarial_result_text(
            "synthetic_communication_decoy",
            {
                "status": "evaluated_frozen_benign_only_baseline",
                "target_is_training": False,
                "false_positive_rate": 0.5,
            },
        )
        == "synthetic_communication_decoy: false-positive rate 0.5"
    )
    assert (
        _adversarial_result_text(
            "diloco_inspired",
            {"status": "sealed_final_holdout"},
        )
        == "diloco_inspired: sealed final holdout (not scored)"
    )
