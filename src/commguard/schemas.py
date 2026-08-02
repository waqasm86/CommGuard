"""Versioned, dependency-free artifact schemas and validation."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from commguard.exceptions import ValidationError

LEGACY_SCHEMA_VERSION = "1.0"
CURRENT_SCHEMA_VERSION = "2.0"
# Compatibility constant for callers that construct telemetry and summary
# envelopes directly. New research manifests use CURRENT_SCHEMA_VERSION.
SCHEMA_VERSION = LEGACY_SCHEMA_VERSION
SUPPORTED_SCHEMA_VERSIONS = frozenset({LEGACY_SCHEMA_VERSION, CURRENT_SCHEMA_VERSION})
TELEMETRY_FIELDS = (
    "gpu_utilization_pct",
    "memory_utilization_pct",
    "memory_used_bytes",
    "power_draw_w",
    "temperature_c",
    "sm_clock_mhz",
    "memory_clock_mhz",
    "pcie_tx_bytes_per_s",
    "pcie_rx_bytes_per_s",
)
FIELD_UNITS = {
    "gpu_utilization_pct": "percent",
    "memory_utilization_pct": "percent",
    "memory_used_bytes": "bytes",
    "power_draw_w": "watts",
    "temperature_c": "celsius",
    "sm_clock_mhz": "MHz",
    "memory_clock_mhz": "MHz",
    "pcie_tx_bytes_per_s": "bytes/second",
    "pcie_rx_bytes_per_s": "bytes/second",
}
ARTIFACT_KINDS = {
    "environment_report",
    "run_manifest",
    "telemetry_sample",
    "workload_event",
    "feature_row",
    "split_assignment",
    "evaluation_result",
    "experiment_summary",
    "calibration_result",
    "corpus_manifest",
    "coverage_record",
    "feature_extraction_result",
}


def _require(condition: bool, path: str, message: str) -> None:
    if not condition:
        raise ValidationError(f"{path}: {message}")


def _utc_timestamp(value: str, path: str) -> None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{path}: expected an ISO-8601 timestamp") from exc
    _require(parsed.tzinfo is not None, path, "timestamp must include a timezone")


def _supported_schema(value: str, path: str = "schema_version") -> None:
    _require(value in SUPPORTED_SCHEMA_VERSIONS, path, "unsupported version")


@dataclass(frozen=True)
class FieldReading:
    """One NVML measurement with explicit support and error state."""

    value: float | int | None
    unit: str
    supported: bool
    error: str | None = None

    def validate(self, path: str = "field") -> None:
        _require(bool(self.unit), f"{path}.unit", "must be non-empty")
        if self.supported:
            _require(self.error is None, f"{path}.error", "must be null when supported")
            _require(self.value is not None, f"{path}.value", "must be present when supported")
            _require(
                isinstance(self.value, (int, float)) and math.isfinite(float(self.value)),
                f"{path}.value",
                "must be a finite number",
            )
        else:
            _require(self.value is None, f"{path}.value", "unsupported values must be null")
            _require(bool(self.error), f"{path}.error", "unsupported values require a reason")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FieldReading:
        _require(
            isinstance(data.get("supported"), bool),
            "field.supported",
            "must be a boolean",
        )
        item = cls(
            value=data.get("value"),
            unit=str(data.get("unit", "")),
            supported=bool(data.get("supported", False)),
            error=data.get("error"),
        )
        item.validate()
        return item


@dataclass(frozen=True)
class TelemetrySample:
    """One per-GPU sample from the host-local collector."""

    run_id: str
    gpu_index: int
    gpu_uuid: str
    sequence: int
    wall_time_utc: str
    monotonic_ns: int
    fields: Mapping[str, FieldReading]
    schema_version: str = SCHEMA_VERSION
    artifact_kind: str = "telemetry_sample"

    def validate(self) -> None:
        _supported_schema(self.schema_version)
        _require(bool(self.run_id), "run_id", "must be non-empty")
        _require(self.gpu_index >= 0, "gpu_index", "must be non-negative")
        _require(self.sequence >= 0, "sequence", "must be non-negative")
        _require(self.monotonic_ns >= 0, "monotonic_ns", "must be non-negative")
        _utc_timestamp(self.wall_time_utc, "wall_time_utc")
        actual = set(self.fields)
        expected = set(TELEMETRY_FIELDS)
        _require(actual == expected, "fields", f"expected exactly {sorted(expected)}")
        for name, reading in self.fields.items():
            reading.validate(f"fields.{name}")
            _require(reading.unit == FIELD_UNITS[name], f"fields.{name}.unit", "unexpected unit")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TelemetrySample:
        item = cls(
            run_id=str(data.get("run_id", "")),
            gpu_index=int(data.get("gpu_index", -1)),
            gpu_uuid=str(data.get("gpu_uuid", "")),
            sequence=int(data.get("sequence", -1)),
            wall_time_utc=str(data.get("wall_time_utc", "")),
            monotonic_ns=int(data.get("monotonic_ns", -1)),
            fields={
                name: FieldReading.from_dict(value)
                for name, value in dict(data.get("fields", {})).items()
            },
            schema_version=str(data.get("schema_version", "")),
            artifact_kind=str(data.get("artifact_kind", "")),
        )
        item.validate()
        return item


@dataclass(frozen=True)
class WorkloadEvent:
    run_id: str
    rank: int
    local_rank: int
    event: str
    wall_time_utc: str
    monotonic_ns: int
    details: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION
    artifact_kind: str = "workload_event"

    def validate(self) -> None:
        _supported_schema(self.schema_version)
        _require(bool(self.run_id), "run_id", "must be non-empty")
        _require(self.rank >= 0 and self.local_rank >= 0, "rank", "must be non-negative")
        _require(bool(self.event), "event", "must be non-empty")
        _utc_timestamp(self.wall_time_utc, "wall_time_utc")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


@dataclass(frozen=True)
class RunManifest:
    run_id: str
    workload_name: str
    workload_label: str
    workload_family: str
    designation: str
    seed: int
    world_size: int
    config: Mapping[str, Any]
    environment: Mapping[str, Any]
    environment_fingerprint: str
    source_commit: str
    started_at_utc: str
    ended_at_utc: str
    warmup_seconds: float
    exit_status: str
    failure_category: str | None
    failure_reason: str | None
    rank_exit_codes: Mapping[str, int]
    nccl_environment: Mapping[str, str]
    participation_valid: bool
    experiment_session_id: str | None = None
    collection_id: str | None = None
    corpus_id: str | None = None
    node_id: str | None = None
    source_dirty: bool | None = None
    input_archive_sha256: str | None = None
    notebook_version: str | None = None
    random_seed: int | None = None
    measurement_start_monotonic_ns: int | None = None
    measurement_end_monotonic_ns: int | None = None
    measured_duration_seconds: float | None = None
    legacy_grouping_ambiguous: bool = False
    schema_version: str = CURRENT_SCHEMA_VERSION
    artifact_kind: str = "run_manifest"

    def validate(self) -> None:
        _supported_schema(self.schema_version)
        _require(bool(self.run_id), "run_id", "must be non-empty")
        _require(bool(self.environment_fingerprint), "environment_fingerprint", "must be non-empty")
        _require(bool(self.source_commit), "source_commit", "must be non-empty")
        _require(self.warmup_seconds >= 0, "warmup_seconds", "must be non-negative")
        _require(self.world_size == 2, "world_size", "dual-T4 manifests require world size 2")
        _require(
            self.designation in {"benign", "adversarial", "calibration"},
            "designation",
            "invalid",
        )
        _require(self.exit_status in {"completed", "failed", "timeout"}, "exit_status", "invalid")
        _utc_timestamp(self.started_at_utc, "started_at_utc")
        _utc_timestamp(self.ended_at_utc, "ended_at_utc")
        _require(set(self.rank_exit_codes) == {"0", "1"}, "rank_exit_codes", "both ranks required")
        if self.exit_status != "completed":
            _require(bool(self.failure_reason), "failure_reason", "required for failed runs")
        if self.schema_version == CURRENT_SCHEMA_VERSION and not self.legacy_grouping_ambiguous:
            _require(bool(self.experiment_session_id), "experiment_session_id", "required")
            _require(bool(self.collection_id), "collection_id", "required")
            _require(bool(self.corpus_id), "corpus_id", "required")
            _require(bool(self.node_id), "node_id", "required")
            _require(isinstance(self.source_dirty, bool), "source_dirty", "must be a boolean")
            _require(self.random_seed == self.seed, "random_seed", "must equal seed")
            if self.exit_status == "completed" and self.participation_valid:
                _require(
                    self.measurement_start_monotonic_ns is not None,
                    "measurement_start_monotonic_ns",
                    "required for a completed v2 run",
                )
                _require(
                    self.measurement_end_monotonic_ns is not None,
                    "measurement_end_monotonic_ns",
                    "required for a completed v2 run",
                )
                _require(
                    self.measured_duration_seconds is not None
                    and self.measured_duration_seconds >= 0,
                    "measured_duration_seconds",
                    "must be non-negative",
                )
                assert self.measurement_start_monotonic_ns is not None
                assert self.measurement_end_monotonic_ns is not None
                assert self.measured_duration_seconds is not None
                _require(
                    self.measurement_start_monotonic_ns >= 0
                    and self.measurement_end_monotonic_ns >= self.measurement_start_monotonic_ns,
                    "measurement interval",
                    "must be ordered and non-negative",
                )
                _require(
                    math.isclose(
                        self.measured_duration_seconds,
                        (self.measurement_end_monotonic_ns - self.measurement_start_monotonic_ns)
                        / 1e9,
                        rel_tol=0.0,
                        abs_tol=1e-6,
                    ),
                    "measured_duration_seconds",
                    "must equal the monotonic measurement interval",
                )

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RunManifest:
        payload = dict(data)
        payload.pop("source_schema_version", None)
        try:
            item = cls(**payload)
        except (TypeError, ValueError) as exc:
            raise ValidationError(f"run_manifest: invalid fields: {exc}") from exc
        item.validate()
        return item


def validate_artifact(data: Mapping[str, Any]) -> None:
    """Validate common envelope fields and known specialized artifacts."""
    kind = data.get("artifact_kind")
    _require(kind in ARTIFACT_KINDS, "artifact_kind", f"unknown kind {kind!r}")
    _supported_schema(str(data.get("schema_version", "")))
    if kind == "telemetry_sample":
        TelemetrySample.from_dict(data)
    elif kind == "run_manifest":
        RunManifest.from_dict(data)
    elif kind == "workload_event":
        WorkloadEvent(**data).validate()
    elif kind == "environment_report":
        _require_fields(
            data,
            "environment_report",
            ("created_at_utc", "session_fingerprint", "gpus", "readiness", "strict_ready"),
        )
        _utc_timestamp(str(data["created_at_utc"]), "created_at_utc")
        _require(isinstance(data["gpus"], list), "gpus", "must be an array")
        _require(isinstance(data["readiness"], dict), "readiness", "must be an object")
        _require(isinstance(data["strict_ready"], bool), "strict_ready", "must be a boolean")
        if data.get("schema_version") == CURRENT_SCHEMA_VERSION:
            _require_fields(
                data,
                "environment_report",
                (
                    "experiment_session_id",
                    "node_id",
                    "environment_fingerprint",
                    "source_commit",
                    "source_dirty",
                ),
            )
            for name in (
                "experiment_session_id",
                "node_id",
                "environment_fingerprint",
                "source_commit",
            ):
                _require(bool(data[name]), name, "must be non-empty")
            _require(
                isinstance(data["source_dirty"], bool),
                "source_dirty",
                "must be a boolean",
            )
    elif kind == "feature_row":
        _require_fields(
            data,
            "feature_row",
            (
                "run_id",
                "session_fingerprint",
                "target_label",
                "workload_family",
                "window_seconds",
                "window_index",
            ),
        )
        _require(float(data["window_seconds"]) > 0, "window_seconds", "must be positive")
        _require(int(data["window_index"]) >= 0, "window_index", "must be non-negative")
        if data.get("schema_version") == CURRENT_SCHEMA_VERSION:
            _require_fields(
                data,
                "feature_row",
                (
                    "plan_id",
                    "experiment_session_id",
                    "collection_id",
                    "corpus_id",
                    "node_id",
                    "environment_fingerprint",
                    "workload_config_id",
                    "aligned_sample_pairs",
                ),
            )
            for name in (
                "plan_id",
                "experiment_session_id",
                "collection_id",
                "corpus_id",
                "node_id",
                "environment_fingerprint",
                "workload_config_id",
            ):
                _require(bool(data[name]), name, "must be non-empty")
            _require(int(data["aligned_sample_pairs"]) >= 2, "aligned_sample_pairs", "too few")
    elif kind == "split_assignment":
        _require_fields(data, "split_assignment", ("run_id", "split"))
        _require(
            data["split"] in {"train", "validation", "test"},
            "split",
            "must be train, validation, or test",
        )
    elif kind == "evaluation_result":
        _require_fields(
            data,
            "evaluation_result",
            ("created_at_utc", "split_assignments", "leakage_audit", "ablations"),
        )
        _utc_timestamp(str(data["created_at_utc"]), "created_at_utc")
        _require(isinstance(data["ablations"], dict), "ablations", "must be an object")
        if data.get("schema_version") == CURRENT_SCHEMA_VERSION:
            _require_fields(
                data,
                "evaluation_result",
                (
                    "primary_communication_only",
                    "actual_split_strategy",
                    "split_plan",
                    "warnings",
                ),
            )
            _require(
                data["primary_communication_only"].get("window_seconds") == 30.0,
                "primary_communication_only.window_seconds",
                "must be 30 seconds",
            )
            _require(
                data["split_plan"].get("diagnostic_only") is False,
                "split_plan.diagnostic_only",
                "primary evaluation cannot use a diagnostic split",
            )
    elif kind == "calibration_result":
        _require_fields(
            data,
            "calibration_result",
            ("status", "observations", "falsification_reasons"),
        )
        _require(
            data["status"] in {"supported", "partially_supported", "not_supported"},
            "status",
            "must be supported, partially_supported, or not_supported",
        )
        _require(isinstance(data["observations"], list), "observations", "must be an array")
    elif kind == "experiment_summary":
        _require_fields(data, "experiment_summary", ("summary_type",))
    elif kind == "corpus_manifest":
        from commguard.corpus import CorpusManifest

        CorpusManifest.from_dict(data).validate()
    elif kind == "coverage_record":
        from commguard.features.coverage import CoverageRecord

        CoverageRecord(**data).validate()
    elif kind == "feature_extraction_result":
        _require_fields(
            data,
            "feature_extraction_result",
            (
                "corpus_id",
                "selection_mode",
                "requested_window_seconds",
                "planned_run_count",
                "included_run_count",
                "excluded_run_count",
                "feature_row_count",
                "reason_counts",
                "window_policy",
            ),
        )
        _require(
            data["selection_mode"] == "declared_corpus_manifest",
            "selection_mode",
            "implicit selection is prohibited",
        )


def _require_fields(data: Mapping[str, Any], path: str, names: tuple[str, ...]) -> None:
    missing = [name for name in names if name not in data]
    _require(not missing, path, f"missing required fields: {missing}")


def _migrate_legacy_record(data: dict[str, Any]) -> dict[str, Any]:
    """Return an in-memory 2.0 view without altering source evidence."""
    if data.get("schema_version") != LEGACY_SCHEMA_VERSION:
        return data
    migrated = dict(data)
    migrated["source_schema_version"] = LEGACY_SCHEMA_VERSION
    migrated["schema_version"] = CURRENT_SCHEMA_VERSION
    migrated["legacy_grouping_ambiguous"] = True
    kind = migrated.get("artifact_kind")
    if kind == "run_manifest":
        migrated.update(
            {
                "experiment_session_id": None,
                "collection_id": None,
                "corpus_id": None,
                "node_id": None,
                "source_dirty": None,
                "input_archive_sha256": None,
                "notebook_version": None,
                "random_seed": migrated.get("seed"),
                "measurement_start_monotonic_ns": None,
                "measurement_end_monotonic_ns": None,
                "measured_duration_seconds": None,
            }
        )
    elif kind == "feature_row":
        # Historical session_fingerprint values described the environment.
        migrated["environment_fingerprint"] = migrated.get("session_fingerprint")
        migrated["experiment_session_id"] = None
        migrated["collection_id"] = None
        migrated["corpus_id"] = None
        migrated["node_id"] = None
    elif kind == "environment_report":
        migrated["environment_fingerprint"] = migrated.get("session_fingerprint")
        migrated["experiment_session_id"] = None
        migrated["node_id"] = None
    return migrated


def load_artifact(
    path: str | Path, *, migrate_legacy: bool = False
) -> dict[str, Any] | list[dict[str, Any]]:
    """Load evidence, optionally exposing a non-destructive migrated view."""
    source = Path(path)
    if source.suffix == ".jsonl":
        records: list[dict[str, Any]] = []
        for line_no, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValidationError(f"{source}:{line_no}: invalid JSON") from exc
            validate_artifact(item)
            records.append(_migrate_legacy_record(item) if migrate_legacy else item)
        return records
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValidationError(f"{source}: invalid JSON") from exc
    _require(isinstance(data, dict), str(source), "top level must be an object")
    validate_artifact(data)
    return _migrate_legacy_record(data) if migrate_legacy else data
