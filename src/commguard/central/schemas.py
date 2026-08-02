"""Versioned content-agnostic central-monitoring protocol messages."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

PROTOCOL_VERSION = "1.0"


def _utc(value: str, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{field} must include a timezone")
    return parsed


@dataclass(frozen=True)
class TelemetryBatch:
    batch_id: str
    agent_id: str
    node_id: str
    experiment_session_id: str
    sequence: int
    created_at_utc: str
    monotonic_start_ns: int
    monotonic_end_ns: int
    samples: tuple[dict[str, Any], ...]
    previous_batch_id: str | None = None
    signature: str | None = None
    protocol_version: str = PROTOCOL_VERSION
    message_type: str = "telemetry_batch"

    def validate(self) -> None:
        if self.protocol_version != PROTOCOL_VERSION:
            raise ValueError(f"unsupported protocol version {self.protocol_version!r}")
        if self.message_type != "telemetry_batch":
            raise ValueError("message_type must be telemetry_batch")
        for name in ("batch_id", "agent_id", "node_id", "experiment_session_id"):
            if not getattr(self, name):
                raise ValueError(f"{name} must be non-empty")
        if self.sequence < 0:
            raise ValueError("sequence must be non-negative")
        if self.monotonic_start_ns < 0 or self.monotonic_end_ns < self.monotonic_start_ns:
            raise ValueError("batch monotonic interval must be ordered and non-negative")
        if not self.samples:
            raise ValueError("telemetry batch must contain samples")
        _utc(self.created_at_utc, "created_at_utc")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TelemetryBatch:
        payload = dict(data)
        payload["samples"] = tuple(dict(item) for item in payload.get("samples", ()))
        try:
            message = cls(**payload)
        except TypeError as exc:
            raise ValueError(f"invalid telemetry batch fields: {exc}") from exc
        message.validate()
        return message


@dataclass(frozen=True)
class Heartbeat:
    heartbeat_id: str
    agent_id: str
    node_id: str
    experiment_session_id: str
    sequence: int
    created_at_utc: str
    last_batch_id: str | None
    signature: str | None = None
    protocol_version: str = PROTOCOL_VERSION
    message_type: str = "heartbeat"

    def validate(self) -> None:
        if self.protocol_version != PROTOCOL_VERSION or self.message_type != "heartbeat":
            raise ValueError("unsupported heartbeat protocol envelope")
        for name in ("heartbeat_id", "agent_id", "node_id", "experiment_session_id"):
            if not getattr(self, name):
                raise ValueError(f"{name} must be non-empty")
        if self.sequence < 0:
            raise ValueError("sequence must be non-negative")
        _utc(self.created_at_utc, "created_at_utc")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Heartbeat:
        try:
            message = cls(**data)
        except TypeError as exc:
            raise ValueError(f"invalid heartbeat fields: {exc}") from exc
        message.validate()
        return message


@dataclass(frozen=True)
class IngestionAck:
    message_id: str
    accepted: bool
    reason_code: str
    detail: str
    server_time_utc: str
    last_accepted_sequence: int | None
    protocol_version: str = PROTOCOL_VERSION
    message_type: str = "ingestion_ack"

    def to_dict(self) -> dict[str, Any]:
        _utc(self.server_time_utc, "server_time_utc")
        return asdict(self)


@dataclass(frozen=True)
class DetectorDecision:
    decision_id: str
    created_at_utc: str
    score: float | None
    threshold: float
    abstained: bool
    reason: str
    model_name: str
    model_version: str
    evidence_window_ids: tuple[str, ...]
    protocol_version: str = PROTOCOL_VERSION
    message_type: str = "detector_decision"

    def to_dict(self) -> dict[str, Any]:
        _utc(self.created_at_utc, "created_at_utc")
        if not 0 <= self.threshold <= 1:
            raise ValueError("decision threshold must be in [0, 1]")
        if self.score is not None and not 0 <= self.score <= 1:
            raise ValueError("decision score must be in [0, 1]")
        return asdict(self)


@dataclass(frozen=True)
class ProtocolErrorMessage:
    message_id: str | None
    reason_code: str
    detail: str
    server_time_utc: str
    retryable: bool
    protocol_version: str = PROTOCOL_VERSION
    message_type: str = "protocol_error"

    def to_dict(self) -> dict[str, Any]:
        _utc(self.server_time_utc, "server_time_utc")
        return asdict(self)
