"""Evidence-grounded Markdown report generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from commguard.artifacts import sha256_file

NON_AFFILIATION = (
    "This is an independent, unofficial research prototype. It is not affiliated with or "
    "endorsed by SPAR, Kairos, ERA, UChicago XLab, William Fowler, or the authors and "
    "institutions cited in the related literature."
)


def _adversarial_result_text(family: str, result: dict[str, Any]) -> str:
    status = str(result.get("status", "unavailable"))
    if status == "evaluated_frozen_benign_only_baseline":
        if result.get("target_is_training"):
            return f"{family}: evasion rate {result.get('evasion_rate')}"
        return f"{family}: false-positive rate {result.get('false_positive_rate')}"
    if status == "sealed_final_holdout":
        return f"{family}: sealed final holdout (not scored)"
    return f"{family}: {status.replace('_', ' ')}"


def _latest_json(directory: Path, pattern: str) -> dict[str, Any] | None:
    paths = sorted(directory.glob(pattern))
    if not paths:
        return None
    return json.loads(paths[-1].read_text(encoding="utf-8"))


def _latest_path(directory: Path, pattern: str) -> Path | None:
    paths = sorted(directory.glob(pattern))
    return paths[-1] if paths else None


def _metric(value: Any) -> str:
    return "unavailable" if value is None else f"{float(value):.3f}"


def _primary_result(evaluation: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    current = evaluation.get("primary_communication_only", {}).get("metrics")
    if isinstance(current, dict):
        return "communication_only", current
    legacy = evaluation.get("ablations", {}).get("pcie_only")
    if isinstance(legacy, dict):
        return "pcie_only_legacy", legacy
    return None


def _selected_metrics(result: dict[str, Any]) -> tuple[str, dict[str, Any], dict[str, Any]]:
    if "selected_model" in result:
        return (
            str(result["selected_model"]),
            dict(result.get("window_level", {})),
            dict(result.get("run_level", {})),
        )
    model = str(result.get("best_model", "unavailable"))
    return model, dict(result.get(model, {})), dict(result.get("run_level", {}))


def _ablation_summary(name: str, result: dict[str, Any]) -> str:
    model, window, run = _selected_metrics(result)
    return (
        f"- `{name}`: selected `{model}`; window balanced accuracy "
        f"{_metric(window.get('balanced_accuracy'))}; run balanced accuracy "
        f"{_metric(run.get('balanced_accuracy'))}."
    )


def generate_report(
    input_root: str | Path = "artifacts",
    output: str | Path | None = None,
) -> str:
    """Generate a report using only available saved evidence."""
    root = Path(input_root)
    manifest_paths = sorted((root / "runs").glob("*/manifest.json"))
    manifests = [json.loads(path.read_text(encoding="utf-8")) for path in manifest_paths]
    preflight_path = _latest_path(root / "environment", "preflight-*.json")
    calibration_path = _latest_path(root / "results", "calibration-*.json")
    evaluation_path = _latest_path(root / "results", "evaluation-*.json")
    matrix_path = _latest_path(root / "results", "matrix-*.json")
    preflight = _latest_json(root / "environment", "preflight-*.json")
    calibration = _latest_json(root / "results", "calibration-*.json")
    evaluation = _latest_json(root / "results", "evaluation-*.json")
    matrix = _latest_json(root / "results", "matrix-*.json")
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
        "## Corpus coverage",
        "",
    ]
    if matrix and matrix.get("family_counts"):
        lines += [
            "| Family | Planned | Completed | Failed | Primary valid | Excluded |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for family, counts in sorted(matrix["family_counts"].items()):
            lines.append(
                f"| `{family}` | {counts.get('planned', 0)} | "
                f"{counts.get('completed', 0)} | {counts.get('failed', 0)} | "
                f"{counts.get('feature_valid', 0)} | "
                f"{counts.get('feature_excluded', 0)} |"
            )
        gate = matrix.get("primary_coverage_gate", {})
        lines += ["", f"Saved primary coverage gate passed: `{gate.get('passed', False)}`."]
    else:
        lines.append("No saved matrix coverage summary was found; detector coverage is unknown.")
    lines += [
        "",
        "### Split groups",
        "",
    ]
    split_plan = evaluation.get("split_plan") if evaluation else None
    if isinstance(split_plan, dict):
        lines += ["| Split | Run count | Session IDs |", "|---|---:|---|"]
        for split in ("train", "validation", "test"):
            run_ids = split_plan.get("run_ids_by_split", {}).get(split, [])
            sessions = split_plan.get("session_ids_by_split", {}).get(split, [])
            lines.append(f"| {split} | {len(run_ids)} | {', '.join(sessions) or 'none'} |")
        lines += [
            "",
            f"Actual split strategy: `{split_plan.get('actual_strategy', 'unavailable')}`.",
        ]
    else:
        lines.append("No current split-plan artifact was found.")
    lines += [
        "",
        "## Benign results",
        "",
    ]
    primary = _primary_result(evaluation) if evaluation else None
    if primary:
        primary_name, primary_metrics = primary
        model, window_metrics, run_metrics = _selected_metrics(primary_metrics)
        lines += [
            f"**Observed:** The primary `{primary_name}` result selected `{model}`. "
            f"Window balanced accuracy: {_metric(window_metrics.get('balanced_accuracy'))}; "
            f"run balanced accuracy: {_metric(run_metrics.get('balanced_accuracy'))}; "
            f"run false-positive rate: {_metric(run_metrics.get('false_positive_rate'))}; "
            f"run false-negative rate: {_metric(run_metrics.get('false_negative_rate'))}.",
        ]
        per_family = primary_metrics.get("per_family", {})
        if per_family:
            lines += [
                "",
                "| Family | Samples | Balanced accuracy | FPR | FNR |",
                "|---|---:|---:|---:|---:|",
            ]
            for family, metrics in sorted(per_family.items()):
                lines.append(
                    f"| `{family}` | {metrics.get('sample_count', 0)} | "
                    f"{_metric(metrics.get('balanced_accuracy'))} | "
                    f"{_metric(metrics.get('false_positive_rate'))} | "
                    f"{_metric(metrics.get('false_negative_rate'))} |"
                )
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
                _adversarial_result_text(family, result)
                for family, result in evaluation["heldout_adversarial_families"].items()
            )
            + "."
        )
        costs = evaluation.get("adversarial_efficiency_cost", {}).get("strategies", {})
        if costs:
            duration_ratios = {
                family: value.get(
                    "duration_ratio_vs_baseline",
                    value.get("duration_ratio_vs_ddp_full_parameter"),
                )
                for family, value in costs.items()
            }
            lines.append(
                "**Observed:** Median wall-duration ratios versus ordinary DDP: "
                + "; ".join(f"{family}: {ratio}" for family, ratio in duration_ratios.items())
                + ". These are coarse duration costs, not FLOP efficiency."
            )
    else:
        lines.append(
            "Bounded variants are manifested separately from benign workloads. No "
            "adversarial performance claim is made without a saved evaluation artifact."
        )
    lines += ["", "## Ablations", ""]
    if evaluation and evaluation.get("ablations"):
        for name, result in evaluation["ablations"].items():
            lines.append(_ablation_summary(name, result))
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
        "### Input artifact hashes",
        "",
    ]
    evidence_paths = [
        path
        for path in [
            preflight_path,
            calibration_path,
            evaluation_path,
            matrix_path,
            *manifest_paths,
        ]
        if path is not None
    ]
    if evidence_paths:
        for path in evidence_paths:
            lines.append(f"- `{path.relative_to(root)}`: `{sha256_file(path)}`")
    else:
        lines.append("No input artifacts were found to hash.")
    lines.append("")
    report = "\n".join(lines)
    if output is not None:
        target = Path(output)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise FileExistsError(f"refusing to overwrite report: {target}")
        target.write_text(report, encoding="utf-8")
    return report
