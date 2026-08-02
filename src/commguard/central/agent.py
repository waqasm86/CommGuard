"""Bounded node agent with a pluggable backend and buffered retry."""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any, Protocol

from commguard.central.schemas import Heartbeat, IngestionAck, TelemetryBatch
from commguard.central.security import sign_message, validate_telemetry_samples
from commguard.schemas import TelemetrySample


class AgentBackend(Protocol):
    def collect(self) -> tuple[TelemetrySample | dict[str, Any], ...]: ...


class AgentTransport(Protocol):
    def send(self, message: dict[str, Any]) -> IngestionAck: ...


def protocol_sample(sample: TelemetrySample | dict[str, Any]) -> dict[str, Any]:
    payload = sample.to_dict() if isinstance(sample, TelemetrySample) else dict(sample)
    fields = payload.get("fields")
    if not isinstance(fields, dict):
        raise ValueError("backend sample fields must be an object")
    return {
        "sample_id": str(payload.get("sample_id") or f"sample-{uuid.uuid4().hex}"),
        "observed_at_utc": str(payload.get("observed_at_utc") or payload.get("wall_time_utc")),
        "monotonic_ns": int(payload["monotonic_ns"]),
        "gpu_index": int(payload["gpu_index"]),
        "gpu_uuid": str(payload["gpu_uuid"]),
        "fields": fields,
    }


class NodeAgent:
    def __init__(
        self,
        *,
        agent_id: str,
        node_id: str,
        experiment_session_id: str,
        hmac_secret: bytes,
        backend: AgentBackend,
        transport: AgentTransport,
        clock: Callable[[], datetime] | None = None,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        if not agent_id or not node_id or not experiment_session_id or not hmac_secret:
            raise ValueError("node agent requires identity, session, and HMAC secret")
        self.agent_id = agent_id
        self.node_id = node_id
        self.experiment_session_id = experiment_session_id
        self.hmac_secret = hmac_secret
        self.backend = backend
        self.transport = transport
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.monotonic_ns = monotonic_ns
        self.next_sequence = 0
        self.last_batch_id: str | None = None
        self.pending: list[dict[str, Any]] = []

    def _sign(self, message: TelemetryBatch | Heartbeat) -> dict[str, Any]:
        unsigned = message.to_dict()
        signature = sign_message(unsigned, self.hmac_secret)
        return replace(message, signature=signature).to_dict()

    def collect_once(self) -> str:
        started_ns = self.monotonic_ns()
        samples = tuple(protocol_sample(sample) for sample in self.backend.collect())
        validate_telemetry_samples(samples)
        ended_ns = self.monotonic_ns()
        batch_id = f"batch-{uuid.uuid4().hex}"
        previous_batch_id = (
            str(self.pending[-1]["batch_id"]) if self.pending else self.last_batch_id
        )
        batch = TelemetryBatch(
            batch_id=batch_id,
            agent_id=self.agent_id,
            node_id=self.node_id,
            experiment_session_id=self.experiment_session_id,
            sequence=self.next_sequence,
            created_at_utc=self.clock().isoformat(),
            monotonic_start_ns=started_ns,
            monotonic_end_ns=max(started_ns, ended_ns),
            samples=samples,
            previous_batch_id=previous_batch_id,
        )
        self.pending.append(self._sign(batch))
        self.next_sequence += 1
        return batch_id

    def flush(self) -> list[IngestionAck]:
        acknowledgments: list[IngestionAck] = []
        while self.pending:
            message = self.pending[0]
            acknowledgment = self.transport.send(message)
            acknowledgments.append(acknowledgment)
            if not acknowledgment.accepted:
                break
            self.pending.pop(0)
            self.last_batch_id = str(message["batch_id"])
        return acknowledgments

    def heartbeat(self) -> IngestionAck:
        heartbeat = Heartbeat(
            heartbeat_id=f"heartbeat-{uuid.uuid4().hex}",
            agent_id=self.agent_id,
            node_id=self.node_id,
            experiment_session_id=self.experiment_session_id,
            sequence=self.next_sequence,
            created_at_utc=self.clock().isoformat(),
            last_batch_id=self.last_batch_id,
        )
        self.next_sequence += 1
        return self.transport.send(self._sign(heartbeat))

    def run(
        self,
        cycles: int,
        interval_seconds: float,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if cycles < 1 or interval_seconds < 0:
            raise ValueError("agent cycles must be positive and interval non-negative")
        for index in range(cycles):
            self.collect_once()
            self.flush()
            if index + 1 < cycles:
                sleeper(interval_seconds)
