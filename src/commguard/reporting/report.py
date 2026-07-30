"""Evidence-grounded Markdown report generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

NON_AFFILIATION = (
    "This is an independent, unofficial research prototype. It is not affiliated with or "
    "endorsed by SPAR, Kairos, ERA, UChicago XLab, William Fowler, or the authors and "
    "institutions cited in the related literature."
)


def _latest_json(directory: Path, pattern: str) -> dict[str, Any] | None:
    paths = sorted(directory.glob(pattern))
    if not paths:
        return None
    return json.loads(paths[-1].read_text(encoding="utf-8"))


def generate_report(
    input_root: str | Path = "artifacts",
    output: str | Path | None = None,
) -> str:
    """Generate a report using only available saved evidence."""
    root = Path(input_root)
    manifests = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((root / "runs").glob("*/manifest.json"))
    ]
    preflight = _latest_json(root / "environment", "preflight-*.json")
    calibration = _latest_json(root / "results", "calibration-*.json")
    evaluation = _latest_json(root / "results", "evaluation-*.json")
    completed = [item for item in manifests if item["exit_status"] == "completed"]
    failed = [item for item in manifests if item["exit_status"] != "completed"]
    lines = [
        "# CommGuard Dual-T4 Research Report",
        "",
        f"> {NON_AFFILIATION}",
        "",
        "## Abstract",
        "",
        "This report describes a limited single-host, dual-NVIDIA-T4 experiment using "
        "content-agnostic NVML signals. Quantitative statements below are generated from "
        "the artifact tree; absent evidence is stated as unmeasured.",
        "",
        "## Research question",
        "",
        "Can short telemetry windows distinguish the included standard two-rank PyTorch "
        "DDP training workloads from included benign inference and GPU controls in the "
        "recorded Kaggle sessions?",
        "",
        "## Hypothesis and falsification criteria",
        "",
        "The hypothesis is that communication-correlated PCIe readings and temporal GPU "
        "signals respond consistently enough to controlled collectives to justify a bounded "
        "training-versus-non-training comparison. It is falsified for this session if PCIe "
        "fields are unsupported/constant, fail to respond monotonically, matched classes "
        "substantially overlap, grouped performance collapses, hard negatives create many "
        "false positives, or session holdout does not generalize.",
        "",
        "## Related work and scope difference",
        "",
        "The local bibliography motivates NVML telemetry, communication-size signals, "
        "adversarial iteration, and evidence separation. CommGuard does not reproduce the "
        "cited studies, emulate frontier clusters, or implement treaty verification.",
        "",
        "## Hardware/software environment",
        "",
    ]
    if preflight:
        names = ", ".join(gpu["name"] for gpu in preflight["gpus"])
        lines += [
            f"**Observed:** GPUs: {names or 'none'}. Strict readiness: "
            f"`{preflight['strict_ready']}`. Session fingerprint: "
            f"`{preflight['session_fingerprint']}`.",
        ]
    else:
        lines += ["No saved preflight artifact was found."]
    lines += [
        "",
        "## Workload suite",
        "",
        f"**Observed:** {len(manifests)} manifested runs: {len(completed)} completed and "
        f"{len(failed)} failed or timed out. Families present: "
        + (
            ", ".join(sorted({item["workload_family"] for item in manifests}))
            if manifests
            else "none"
        )
        + ".",
        "",
        "## Telemetry semantics",
        "",
        "Nine fields are attempted per GPU: utilization, memory utilization, memory used, "
        "power, temperature, SM clock, memory clock, PCIe TX, and PCIe RX. Unsupported "
        "readings remain null with an error. PCIe TX/RX are PCIe traffic readings, not a "
        "complete or direct measure of NCCL bytes.",
        "",
        "## Dataset construction",
        "",
        "Raw telemetry and manifests are create-only. Derived windows retain run/session "
        "group identity only for splitting and never expose it to classifiers.",
        "",
        "## Leakage controls",
        "",
        "All windows from one run remain in one split. Session holdout is used when enough "
        "sessions exist. IDs, timestamps, paths, labels, workload family, and duration "
        "metadata are excluded from model features.",
        "",
        "## Feature engineering",
        "",
        "Features include distribution statistics, slopes, autocorrelation, duty cycle, "
        "PCIe ratios/totals, missingness, and cross-GPU divergence/correlation at multiple "
        "window lengths.",
        "",
        "## Evaluation protocol",
        "",
        "Majority, a fitted low-dimensional PCIe rule, logistic regression, and random "
        "forest are compared. PCIe-only, non-PCIe, and combined ablations are separate.",
        "",
        "## Benign results",
        "",
    ]
    if evaluation:
        combined = evaluation["ablations"]["combined"]
        best = combined["best_model"]
        metrics = combined[best]
        lines += [
            f"**Observed:** The combined-signal best baseline was `{best}` on "
            f"{metrics['sample_count']} held-out windows: balanced accuracy "
            f"{metrics['balanced_accuracy']:.3f}, F1 {metrics['f1']:.3f}, false-positive "
            f"rate {metrics['false_positive_rate']}, and false-negative rate "
            f"{metrics['false_negative_rate']}. Run-level results and per-family metrics "
            "are stored in the evaluation artifact.",
        ]
    else:
        lines += ["No saved detector evaluation was found; no performance claim is made."]
    lines += [
        "",
        "## Adversarial results",
        "",
    ]
    if evaluation and evaluation.get("heldout_adversarial_families"):
        lines.append(
            "**Observed:** Complete-strategy holdouts: "
            + "; ".join(
                (
                    f"{family}: detection "
                    f"{result.get('training_detection_rate', 'unmeasured')}"
                    if result.get("status") == "evaluated"
                    else f"{family}: insufficient data"
                )
                for family, result in evaluation["heldout_adversarial_families"].items()
            )
            + "."
        )
        costs = evaluation.get("adversarial_efficiency_cost", {}).get("strategies", {})
        if costs:
            lines.append(
                "**Observed:** Median wall-duration ratios versus ordinary DDP: "
                + "; ".join(
                    f"{family}: {value['duration_ratio_vs_ddp_full_parameter']}"
                    for family, value in costs.items()
                )
                + ". These are coarse duration costs, not FLOP efficiency."
            )
    else:
        lines.append(
            "Bounded variants are manifested separately from benign workloads. No "
            "adversarial performance claim is made without a saved evaluation artifact."
        )
    lines += ["", "## Ablations", ""]
    if evaluation:
        for name, result in evaluation["ablations"].items():
            best = result["best_model"]
            lines.append(
                f"- `{name}`: best `{best}`, balanced accuracy "
                f"{result[best]['balanced_accuracy']:.3f}."
            )
    else:
        lines.append("No saved ablation results were found.")
    lines += [
        "",
        "## Failures and negative findings",
        "",
    ]
    if calibration:
        lines += [
            f"**Observed:** Calibration status was `{calibration['status']}`. "
            f"Falsification reasons: "
            f"{'; '.join(calibration['falsification_reasons']) or 'none recorded'}.",
        ]
    else:
        lines += ["Calibration has not been recorded; detector escalation is unsupported."]
    if failed:
        lines.append(
            "**Observed:** Failed categories: "
            + ", ".join(
                f"{item['failure_category'] or 'unknown'} ({item['run_id']})" for item in failed
            )
            + "."
        )
    lines += [
        "",
        "## Limitations",
        "",
        "This is a single-node, two-GPU, T4, PCIe, Kaggle, small-model study. Accessible "
        "signals can include host transfers and other confounders. The study does not cover "
        "frontier models, multi-node topology, NVLink/NVSwitch, RoCE/InfiniBand, telemetry "
        "integrity, privacy guarantees, determined operators, or deployment security.",
        "",
        "## Implications for future multi-node work",
        "",
        "**Hypothesized:** After successful repeated dual-T4 calibration, a separate "
        "multi-node design could add node-local collectors and NIC evidence. No transfer is "
        "assumed; it would require fresh calibration and threat modeling.",
        "",
        "## Reproducibility instructions",
        "",
        "Install locally with `--no-deps`, run strict preflight, smoke and calibration, then "
        "the bounded profile. Preserve the full artifact tree and repeat in another session.",
        "",
        "## Provenance",
        "",
        "Run manifests embed source commit and environment fingerprints. Headline values in "
        "this report are read from saved preflight, manifest, calibration, and evaluation "
        "artifacts.",
        "",
    ]
    report = "\n".join(lines)
    if output is not None:
        target = Path(output)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise FileExistsError(f"refusing to overwrite report: {target}")
        target.write_text(report, encoding="utf-8")
    return report
