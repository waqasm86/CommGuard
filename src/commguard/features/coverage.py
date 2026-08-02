"""Machine-readable feature coverage diagnostics and strict primary gates."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from commguard.exceptions import CoverageError, ValidationError
from commguard.schemas import CURRENT_SCHEMA_VERSION, load_artifact
from commguard.workloads import BENIGN_REQUIRED_FAMILIES

PRIMARY_BENIGN_FAMILIES = BENIGN_REQUIRED_FAMILIES
PRIMARY_WINDOW_SECONDS = (30.0,)
DIAGNOSTIC_WINDOW_SECONDS = (5.0, 15.0)
DEFAULT_WINDOW_SECONDS = (*DIAGNOSTIC_WINDOW_SECONDS, *PRIMARY_WINDOW_SECONDS)


class CoverageReason(str, Enum):
    NOT_IN_SELECTED_CORPUS = "not_in_selected_corpus"
    INVALID_MANIFEST = "invalid_manifest"
    INCOMPLETE_RUN = "incomplete_run"
    PARTICIPATION_INVALID = "participation_invalid"
    CALIBRATION_RUN_EXCLUDED = "calibration_run_excluded"
    MISSING_TELEMETRY = "missing_telemetry"
    MISSING_REQUIRED_GPU = "missing_required_gpu"
    NO_COMMON_INTERVAL = "no_common_interval"
    POST_WARMUP_INTERVAL_TOO_SHORT = "post_warmup_interval_too_short"
    INSUFFICIENT_SAMPLES = "insufficient_samples"
    UNSUPPORTED_TELEMETRY_SCHEMA = "unsupported_telemetry_schema"
    NO_COMMON_WINDOW_LENGTH = "no_common_window_length"
    FEATURE_ERROR = "feature_error"


REQUIRED_REASON_CODES = frozenset(reason.value for reason in CoverageReason)


@dataclass(frozen=True)
class CoverageRecord:
    plan_id: str
    run_id: str | None
    workload_family: str
    target_label: str
    designation: str
    experiment_session_id: str
    collection_id: str
    corpus_id: str
    node_id: str
    status: str
    reason_code: str | None
    detail: str
    started_at_utc: str | None
    ended_at_utc: str | None
    rows_per_gpu: dict[str, int]
    common_start_monotonic_ns: int | None
    common_end_monotonic_ns: int | None
    common_overlap_seconds: float
    configured_warmup_seconds: float
    usable_duration_seconds: float
    requested_window_seconds: tuple[float, ...]
    emitted_windows: dict[str, int]
    sampling_gap_seconds_by_gpu: dict[str, dict[str, float | int | None]]
    alignment_tolerance_seconds: float
    aligned_sample_pairs: int
    legacy_grouping_ambiguous: bool
    schema_version: str = CURRENT_SCHEMA_VERSION
    artifact_kind: str = "coverage_record"

    def validate(self) -> None:
        required_context = {
            "plan_id": self.plan_id,
            "workload_family": self.workload_family,
            "target_label": self.target_label,
            "experiment_session_id": self.experiment_session_id,
            "collection_id": self.collection_id,
            "corpus_id": self.corpus_id,
            "node_id": self.node_id,
        }
        missing_context = sorted(name for name, value in required_context.items() if not value)
        if missing_context:
            raise ValidationError(f"coverage record missing context: {missing_context}")
        if self.status not in {"included", "excluded"}:
            raise ValidationError(f"coverage {self.plan_id}: invalid status {self.status!r}")
        if self.status == "included" and self.reason_code is not None:
            raise ValidationError(f"coverage {self.plan_id}: included record cannot have a reason")
        if self.status == "excluded" and self.reason_code not in REQUIRED_REASON_CODES:
            raise ValidationError(
                f"coverage {self.plan_id}: excluded record requires a supported reason code"
            )
        if not self.requested_window_seconds or any(
            value <= 0 for value in self.requested_window_seconds
        ):
            raise ValidationError(f"coverage {self.plan_id}: requested windows must be positive")
        if self.common_overlap_seconds < 0 or self.usable_duration_seconds < 0:
            raise ValidationError(f"coverage {self.plan_id}: durations cannot be negative")
        if self.usable_duration_seconds > self.common_overlap_seconds + 1e-9:
            raise ValidationError(
                f"coverage {self.plan_id}: usable duration exceeds common overlap"
            )
        if self.configured_warmup_seconds < 0 or self.alignment_tolerance_seconds < 0:
            raise ValidationError(f"coverage {self.plan_id}: configuration cannot be negative")
        if set(self.rows_per_gpu) != {"0", "1"} or any(
            not isinstance(value, int) or value < 0 for value in self.rows_per_gpu.values()
        ):
            raise ValidationError(
                f"coverage {self.plan_id}: rows_per_gpu requires non-negative GPU 0/1 counts"
            )
        requested_keys = {f"{window:g}" for window in self.requested_window_seconds}
        if set(self.emitted_windows) != requested_keys or any(
            not isinstance(value, int) or value < 0 for value in self.emitted_windows.values()
        ):
            raise ValidationError(
                f"coverage {self.plan_id}: emitted windows must match requested windows"
            )
        if set(self.sampling_gap_seconds_by_gpu) != {"0", "1"}:
            raise ValidationError(f"coverage {self.plan_id}: sampling gaps require GPU 0 and GPU 1")
        for gpu, gap_stats in self.sampling_gap_seconds_by_gpu.items():
            for name, value in gap_stats.items():
                if value is not None and value < 0:
                    raise ValidationError(
                        f"coverage {self.plan_id}: GPU {gpu} sampling gap {name} is negative"
                    )
        if self.aligned_sample_pairs < 0:
            raise ValidationError(
                f"coverage {self.plan_id}: aligned sample pairs cannot be negative"
            )
        if (self.common_start_monotonic_ns is None) != (self.common_end_monotonic_ns is None):
            raise ValidationError(
                f"coverage {self.plan_id}: common interval endpoints must appear together"
            )
        if (
            self.common_start_monotonic_ns is not None
            and self.common_end_monotonic_ns is not None
            and self.common_end_monotonic_ns < self.common_start_monotonic_ns
        ):
            raise ValidationError(f"coverage {self.plan_id}: common interval is reversed")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


@dataclass(frozen=True)
class ExtractionResult:
    features: tuple[dict[str, Any], ...]
    coverage: tuple[CoverageRecord, ...]
    corpus_id: str
    requested_window_seconds: tuple[float, ...]
    selection_mode: str = "declared_corpus_manifest"
    selected_designations: tuple[str, ...] = ("benign",)
    calibration_reference: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        for record in self.coverage:
            record.validate()
        plan_ids = [record.plan_id for record in self.coverage]
        if len(plan_ids) != len(set(plan_ids)):
            raise ValidationError("coverage result must contain one record per unique plan ID")
        if any(record.corpus_id != self.corpus_id for record in self.coverage):
            raise ValidationError("coverage result contains a different corpus ID")
        if any(
            record.requested_window_seconds != self.requested_window_seconds
            for record in self.coverage
        ):
            raise ValidationError("coverage result contains a different requested-window policy")
        if (
            not self.selected_designations
            or len(self.selected_designations) != len(set(self.selected_designations))
            or any(
                value not in {"benign", "adversarial", "calibration"}
                for value in self.selected_designations
            )
        ):
            raise ValidationError("coverage result has invalid selected designations")

    def summary(self) -> dict[str, Any]:
        reasons = Counter(
            record.reason_code for record in self.coverage if record.reason_code is not None
        )
        return {
            "artifact_kind": "feature_extraction_result",
            "schema_version": CURRENT_SCHEMA_VERSION,
            "corpus_id": self.corpus_id,
            "selection_mode": self.selection_mode,
            "selected_designations": list(self.selected_designations),
            "calibration_reference": self.calibration_reference,
            "requested_window_seconds": list(self.requested_window_seconds),
            "planned_run_count": len(self.coverage),
            "included_run_count": sum(record.status == "included" for record in self.coverage),
            "excluded_run_count": sum(record.status == "excluded" for record in self.coverage),
            "feature_row_count": len(self.features),
            "reason_counts": dict(sorted(reasons.items())),
            "window_policy": {
                "primary_seconds": list(PRIMARY_WINDOW_SECONDS),
                "diagnostic_short_seconds": list(DIAGNOSTIC_WINDOW_SECONDS),
                "requested_seconds": list(self.requested_window_seconds),
                "primary_window_is_never_implicitly_reduced": True,
            },
        }


def load_extraction_result(
    input_root: str | Path,
    summary_path: str | Path | None = None,
) -> ExtractionResult:
    """Load one linked feature/coverage result from an artifact root.

    ``summary_path`` may be an artifact-root-relative path or an absolute path.
    The newest-summary behavior is retained only for callers that omit it.
    """
    root = Path(input_root)
    if summary_path is None:
        summaries = sorted((root / "features").glob("extraction-*.json"))
        if not summaries:
            raise CoverageError(f"artifact_root={root} has no feature extraction coverage summary")
        resolved_summary_path = summaries[-1]
    else:
        candidate = Path(summary_path)
        resolved_summary_path = candidate if candidate.is_absolute() else root / candidate
        try:
            resolved_summary_path.resolve().relative_to(root.resolve())
        except ValueError as exc:
            raise CoverageError(
                f"extraction_summary={resolved_summary_path} is outside artifact_root={root}"
            ) from exc
        if not resolved_summary_path.is_file():
            raise CoverageError(
                f"artifact_root={root} has no extraction_summary={resolved_summary_path}"
            )
    summary = load_artifact(resolved_summary_path)
    assert isinstance(summary, dict)
    feature_rows = load_artifact(root / str(summary["feature_artifact"]))
    coverage_rows = load_artifact(root / str(summary["coverage_artifact"]))
    if not isinstance(feature_rows, list) or not isinstance(coverage_rows, list):
        raise ValidationError("feature extraction artifacts must be JSONL arrays")
    records = tuple(
        CoverageRecord(
            **{
                **row,
                "requested_window_seconds": tuple(row["requested_window_seconds"]),
            }
        )
        for row in coverage_rows
    )
    return ExtractionResult(
        features=tuple(feature_rows),
        coverage=records,
        corpus_id=str(summary["corpus_id"]),
        requested_window_seconds=tuple(
            float(value) for value in summary["requested_window_seconds"]
        ),
        selection_mode=str(summary["selection_mode"]),
        selected_designations=tuple(summary.get("selected_designations", ("benign",))),
        calibration_reference=summary.get("calibration_reference"),
    )


def require_primary_coverage(
    result: ExtractionResult,
    *,
    required_families: tuple[str, ...],
    minimum_runs_per_family: int = 1,
    required_window_seconds: tuple[float, ...] | None = None,
) -> dict[str, Any]:
    """Fail unless every required family has enough complete primary-window runs."""
    if minimum_runs_per_family < 1:
        raise ValueError("minimum_runs_per_family must be at least one")
    windows = required_window_seconds or PRIMARY_WINDOW_SECONDS
    if not windows:
        raise ValueError("at least one required window length is required")

    valid_by_family: dict[str, list[str]] = defaultdict(list)
    failures_by_family: dict[str, list[str]] = defaultdict(list)
    for record in result.coverage:
        if record.workload_family not in required_families:
            continue
        has_windows = all(record.emitted_windows.get(f"{window:g}", 0) > 0 for window in windows)
        if record.status == "included" and has_windows and record.run_id is not None:
            valid_by_family[record.workload_family].append(record.run_id)
        else:
            reason = record.reason_code or CoverageReason.NO_COMMON_WINDOW_LENGTH.value
            failures_by_family[record.workload_family].append(f"{record.plan_id}:{reason}")

    missing = {
        family: {
            "valid_runs": len(valid_by_family[family]),
            "required_runs": minimum_runs_per_family,
            "failures": failures_by_family[family],
        }
        for family in required_families
        if len(valid_by_family[family]) < minimum_runs_per_family
    }
    if missing:
        raise CoverageError(f"corpus={result.corpus_id} primary coverage incomplete: {missing}")
    return {
        "corpus_id": result.corpus_id,
        "required_families": list(required_families),
        "minimum_runs_per_family": minimum_runs_per_family,
        "required_window_seconds": list(windows),
        "valid_run_ids_by_family": {
            family: sorted(valid_by_family[family]) for family in required_families
        },
        "passed": True,
    }
