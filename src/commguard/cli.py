"""CommGuard command-line interface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from commguard.artifacts import ArtifactStore, sha256_file
from commguard.corpus import CorpusManifest
from commguard.environment.preflight import check_environment, summarize_environment
from commguard.evaluation import evaluate_detector
from commguard.exceptions import CommGuardError
from commguard.features import extract_features
from commguard.orchestrator import (
    estimate_matrix,
    plan_matrix,
    run_calibration_sweep,
    run_experiment,
    run_matrix,
)
from commguard.reporting import generate_report
from commguard.workloads import list_workloads


def _json(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, default=str))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="commguard",
        description="Dual-T4 content-agnostic workload signal research SDK",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight", help="record environment readiness")
    preflight.add_argument("--strict", action="store_true")
    preflight.add_argument("--check-network", action="store_true")
    preflight.add_argument("--output", type=Path)
    preflight.add_argument("--json", action="store_true")

    subparsers.add_parser("workloads", help="list bounded workload configurations")

    calibrate = subparsers.add_parser("calibrate", help="run collective payload calibration")
    calibrate.add_argument("--output", type=Path, default=Path("artifacts"))
    calibrate.add_argument("--payload-mib", type=int, nargs="+", default=[1, 4, 16, 64, 128])
    calibrate.add_argument("--collective", default="all_reduce")
    calibrate.add_argument("--repetitions", type=int, default=5)
    calibrate.add_argument(
        "--sampling-interval",
        type=float,
        choices=(1.0, 0.5, 0.2),
        default=0.5,
    )
    calibrate.add_argument("--timeout", type=float, default=180)

    run = subparsers.add_parser("run", help="run one workload or a profile")
    target = run.add_mutually_exclusive_group(required=True)
    target.add_argument("--workload")
    target.add_argument("--profile", choices=("smoke", "standard", "extended"))
    run.add_argument("--output", type=Path, default=Path("artifacts"))
    run.add_argument("--repetitions", type=int)
    run.add_argument("--timeout", type=float, default=180)
    run.add_argument("--negative-calibration-mode", action="store_true")
    run.add_argument(
        "--approve-bounded-redteam",
        action="store_true",
        help="explicitly approve one reviewed adversarial workload (never implied by a profile)",
    )

    estimate = subparsers.add_parser("estimate", help="estimate profile cost without running GPUs")
    estimate.add_argument("--profile", choices=("smoke", "standard", "extended"), default="smoke")
    estimate.add_argument("--repetitions", type=int, default=1)

    features = subparsers.add_parser("features", help="derive deterministic feature windows")
    features.add_argument("--input", type=Path, default=Path("artifacts"))
    features.add_argument("--output", type=Path, default=Path("artifacts"))
    features.add_argument("--corpus-manifest", type=Path, required=True)
    features.add_argument(
        "--windows",
        type=float,
        nargs="+",
        default=[5, 15, 30],
        help="requested windows; 30 seconds is primary and 5/15 are diagnostic",
    )
    features.add_argument("--alignment-tolerance-seconds", type=float, default=0.25)
    features.add_argument("--include-startup", action="store_true")

    evaluate = subparsers.add_parser("evaluate", help="run grouped detector evaluation")
    evaluate.add_argument("--input", type=Path, default=Path("artifacts"))
    evaluate.add_argument("--output", type=Path, default=Path("artifacts"))
    evaluate.add_argument("--negative-calibration-mode", action="store_true")
    evaluate.add_argument("--minimum-runs-per-family", type=int, default=3)

    report = subparsers.add_parser("report", help="generate an evidence-grounded report")
    report.add_argument("--input", type=Path, default=Path("artifacts"))
    report.add_argument("--output", type=Path, required=True)

    export = subparsers.add_parser("export", help="archive the complete artifact tree")
    export.add_argument("--input", type=Path, default=Path("artifacts"))
    export.add_argument("--output", type=Path, required=True)

    corpus = subparsers.add_parser("corpus", help="plan or validate a corpus without GPUs")
    corpus_commands = corpus.add_subparsers(dest="corpus_command", required=True)
    corpus_plan = corpus_commands.add_parser("plan", help="print a deterministic run plan")
    corpus_plan.add_argument(
        "--profile",
        choices=("smoke", "standard", "extended"),
        default="standard",
    )
    corpus_plan.add_argument("--repetitions", type=int, default=3)
    corpus_validate = corpus_commands.add_parser("validate", help="validate a corpus manifest")
    corpus_validate.add_argument("manifest", type=Path)

    verify = subparsers.add_parser("verify-artifact", help="verify an artifact SHA-256")
    verify.add_argument("artifact", type=Path)
    verify.add_argument("--sha256", required=True)

    bundle = subparsers.add_parser(
        "build-review-bundle",
        help="create a deterministic non-destructive review archive and checksum",
    )
    bundle.add_argument("--input", type=Path, default=Path("artifacts"))
    bundle.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "preflight":
            result = check_environment(
                strict=args.strict,
                output=args.output,
                check_network=args.check_network,
            )
            _json(result) if args.json else print(summarize_environment(result))
        elif args.command == "workloads":
            _json(list_workloads())
        elif args.command == "calibrate":
            _json(
                run_calibration_sweep(
                    output=args.output,
                    payload_mib=tuple(args.payload_mib),
                    collective=args.collective,
                    repetitions=args.repetitions,
                    timeout_s=args.timeout,
                    sampling_interval_s=args.sampling_interval,
                )
            )
        elif args.command == "run":
            if args.workload:
                result = run_experiment(
                    args.workload,
                    output=args.output,
                    timeout_s=args.timeout,
                    adversarial_approval=args.approve_bounded_redteam,
                )
                _json({"run_id": result["run_id"], "manifest": result["manifest"]})
            else:
                _json(
                    run_matrix(
                        profile=args.profile,
                        output=args.output,
                        repetitions=args.repetitions,
                        negative_calibration_mode=args.negative_calibration_mode,
                        timeout_s=args.timeout,
                    )
                )
        elif args.command == "estimate":
            _json(estimate_matrix(args.profile, args.repetitions))
        elif args.command == "features":
            rows = extract_features(
                input_root=args.input,
                output=args.output,
                window_lengths=args.windows,
                exclude_startup=not args.include_startup,
                corpus_manifest=args.corpus_manifest,
                alignment_tolerance_seconds=args.alignment_tolerance_seconds,
            )
            _json({"feature_rows": len(rows), "output": str(args.output)})
        elif args.command == "evaluate":
            _json(
                evaluate_detector(
                    input_root=args.input,
                    output=args.output,
                    negative_calibration_mode=args.negative_calibration_mode,
                    minimum_runs_per_family=args.minimum_runs_per_family,
                )
            )
        elif args.command == "report":
            generate_report(input_root=args.input, output=args.output)
            print(args.output)
        elif args.command == "export":
            print(ArtifactStore(args.input).export(args.output))
        elif args.command == "corpus":
            if args.corpus_command == "plan":
                plans = plan_matrix(args.profile, args.repetitions)
                _json(
                    {
                        "profile": args.profile,
                        "repetitions": args.repetitions,
                        "development_dry_run_only": True,
                        "scientific_gpu_evidence": False,
                        "planned_runs": [vars(plan) for plan in plans],
                    }
                )
            else:
                data = json.loads(args.manifest.read_text(encoding="utf-8"))
                manifest = CorpusManifest.from_dict(data)
                manifest.validate()
                _json(
                    {
                        "valid": True,
                        "corpus_id": manifest.corpus_id,
                        "planned_runs": len(manifest.planned_runs),
                        "accepted_runs": len(manifest.accepted_run_ids),
                    }
                )
        elif args.command == "verify-artifact":
            actual = sha256_file(args.artifact)
            if actual != args.sha256:
                raise ValueError(f"SHA-256 mismatch: expected={args.sha256} actual={actual}")
            _json({"verified": True, "artifact": str(args.artifact), "sha256": actual})
        elif args.command == "build-review-bundle":
            archive, checksum = ArtifactStore(args.input).export_with_checksum(args.output)
            _json(
                {
                    "archive": str(archive),
                    "checksum_file": str(checksum),
                    "sha256": sha256_file(archive),
                    "source_outputs_preserved": True,
                }
            )
        return 0
    except (CommGuardError, ValueError, RuntimeError, OSError) as exc:
        print(f"commguard: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
