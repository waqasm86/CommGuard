"""Central ingestion, node-health, aggregation, and detector service interfaces."""

from __future__ import annotations

import hashlib
import statistics
import uuid
from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from commguard.central.schemas import (
    PROTOCOL_VERSION,
    DetectorDecision,
    Heartbeat,
    IngestionAck,
    TelemetryBatch,
)
from commguard.central.security import (
    MAX_PAYLOAD_BYTES,
    canonical_bytes,
    validate_payload_size,
    validate_telemetry_samples,
    verify_message,
)
from commguard.schemas import TELEMETRY_FIELDS


@dataclass(frozen=True)
class AgentRegistration:
    agent_id: str
    node_id: str
    hmac_secret: bytes

    def validate(self) -> None:
        if (
            not self.agent_id
            or not self.node_id
            or not isinstance(self.hmac_secret, bytes)
            or not self.hmac_secret
        ):
            raise ValueError("agent registration requires agent/node identity and HMAC secret")


class CentralIngestionService:
    """In-memory reference service; persistence is delegated to a transport."""

    def __init__(
        self,
        registrations: Mapping[str, AgentRegistration],
        *,
        experiment_session_id: str,
        expected_nodes: tuple[str, ...],
        maximum_payload_bytes: int = MAX_PAYLOAD_BYTES,
        maximum_clock_skew_seconds: float = 30.0,
        stale_after_seconds: float = 15.0,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.registrations = dict(registrations)
        for agent_id, registration in self.registrations.items():
            registration.validate()
            if agent_id != registration.agent_id:
                raise ValueError("registration mapping key must equal agent_id")
        if set(expected_nodes) != {item.node_id for item in self.registrations.values()}:
            raise ValueError("expected nodes must exactly match registered node IDs")
        if not experiment_session_id:
            raise ValueError("experiment_session_id is required")
        if maximum_payload_bytes < 1:
            raise ValueError("maximum_payload_bytes must be positive")
        if maximum_clock_skew_seconds < 0 or stale_after_seconds <= 0:
            raise ValueError("clock skew and staleness limits must be non-negative/positive")
        self.experiment_session_id = experiment_session_id
        self.expected_nodes = tuple(sorted(expected_nodes))
        self.expected_agents = tuple(sorted(self.registrations))
        self.maximum_payload_bytes = maximum_payload_bytes
        self.maximum_clock_skew_seconds = maximum_clock_skew_seconds
        self.stale_after_seconds = stale_after_seconds
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.last_sequence_by_agent: dict[str, int] = {}
        self.last_batch_by_agent: dict[str, str] = {}
        self.last_seen_by_node: dict[str, datetime] = {}
        self.last_seen_by_agent: dict[str, datetime] = {}
        self.seen_message_ids: set[str] = set()
        self.accepted_message_digests: dict[str, str] = {}
        self.accepted_samples_by_node: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(
            list
        )
        self.accepted_samples_by_agent: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(
            list
        )

    def _ack(
        self,
        message_id: str,
        accepted: bool,
        reason_code: str,
        detail: str,
        agent_id: str | None = None,
    ) -> IngestionAck:
        return IngestionAck(
            message_id=message_id,
            accepted=accepted,
            reason_code=reason_code,
            detail=detail,
            server_time_utc=self.clock().isoformat(),
            last_accepted_sequence=self.last_sequence_by_agent.get(agent_id or ""),
        )

    @staticmethod
    def _reason(exc: Exception) -> str:
        text = str(exc)
        for code in (
            "payload_too_large",
            "privacy_violation",
            "invalid_sample",
            "unsupported protocol version",
        ):
            if code in text:
                return code.replace(" ", "_")
        return "invalid_message"

    def ingest(self, raw: Mapping[str, Any]) -> IngestionAck:
        payload = dict(raw)
        message_id = str(payload.get("batch_id") or payload.get("heartbeat_id") or "unknown")
        agent_id = str(payload.get("agent_id") or "")
        try:
            validate_payload_size(payload, self.maximum_payload_bytes)
        except ValueError as exc:
            return self._ack(message_id, False, self._reason(exc), str(exc), agent_id)
        if payload.get("protocol_version") != PROTOCOL_VERSION:
            return self._ack(
                message_id,
                False,
                "unsupported_protocol_version",
                f"unsupported protocol version {payload.get('protocol_version')!r}",
                agent_id,
            )
        registration = self.registrations.get(agent_id)
        if registration is None:
            return self._ack(message_id, False, "unknown_agent", "agent is not registered")
        if str(payload.get("node_id")) != registration.node_id:
            return self._ack(
                message_id,
                False,
                "node_identity_mismatch",
                "node does not match agent",
            )
        if not verify_message(payload, registration.hmac_secret):
            return self._ack(message_id, False, "invalid_signature", "HMAC verification failed")
        if payload.get("experiment_session_id") != self.experiment_session_id:
            return self._ack(
                message_id,
                False,
                "session_mismatch",
                "message experiment session does not match the central service",
                agent_id,
            )
        
        # Parse the message based on type
        message: TelemetryBatch | Heartbeat | None = None
        try:
            if payload.get("message_type") == "telemetry_batch":
                message = TelemetryBatch.from_dict(payload)
                # Validate telemetry samples if we have a TelemetryBatch
                validate_telemetry_samples(message.samples)
            elif payload.get("message_type") == "heartbeat":
                message = Heartbeat.from_dict(payload)
            else:
                return self._ack(
                    message_id,
                    False,
                    "unknown_message_type",
                    f"unknown message type {payload.get('message_type')!r}",
                    agent_id,
                )
        except (TypeError, ValueError) as exc:
            return self._ack(message_id, False, self._reason(exc), str(exc), agent_id)

        # Ensure we have a valid message
        if message is None:
            return self._ack(
                message_id,
                False,
                "invalid_message",
                "failed to parse message",
                agent_id,
            )

        authenticated_digest = hashlib.sha256(
            canonical_bytes(payload, include_signature=True)
        ).hexdigest()
        if message_id in self.seen_message_ids:
            if self.accepted_message_digests.get(message_id) == authenticated_digest:
                return self._ack(
                    message_id,
                    True,
                    "already_accepted",
                    "identical authenticated message was already accepted",
                    agent_id,
                )
            return self._ack(
                message_id,
                False,
                "message_id_conflict",
                "message ID was reused with different authenticated content",
                agent_id,
            )
        previous_sequence = self.last_sequence_by_agent.get(agent_id)
        if previous_sequence is not None and message.sequence <= previous_sequence:
            return self._ack(
                message_id,
                False,
                "stale_sequence",
                f"sequence={message.sequence} last_accepted={previous_sequence}",
                agent_id,
            )
        if previous_sequence is not None and message.sequence != previous_sequence + 1:
            return self._ack(
                message_id,
                False,
                "sequence_gap",
                f"sequence={message.sequence} expected={previous_sequence + 1}",
                agent_id,
            )
        now = self.clock()
        created = datetime.fromisoformat(message.created_at_utc.replace("Z", "+00:00"))
        skew = abs((now - created).total_seconds())
        if skew > self.maximum_clock_skew_seconds:
            return self._ack(
                message_id,
                False,
                "clock_skew",
                f"clock skew {skew:.6f}s exceeds {self.maximum_clock_skew_seconds:.6f}s",
                agent_id,
            )
        
        # Handle clock skew validation specifically for TelemetryBatch
        if isinstance(message, TelemetryBatch):
            sample_skews = [
                abs(
                    (
                        now
                        - datetime.fromisoformat(
                            str(sample["observed_at_utc"]).replace("Z", "+00:00")
                        )
                    ).total_seconds()
                )
                for sample in message.samples
            ]
            if sample_skews and max(sample_skews) > self.maximum_clock_skew_seconds:
                return self._ack(
                    message_id,
                    False,
                    "sample_clock_skew",
                    f"sample clock skew {max(sample_skews):.6f}s exceeds limit",
                    agent_id,
                )
            
            # Handle batch chain validation only for TelemetryBatch
            expected_previous = self.last_batch_by_agent.get(agent_id)
            if expected_previous is not None and message.previous_batch_id != expected_previous:
                return self._ack(
                    message_id,
                    False,
                    "batch_chain_mismatch",
                    f"previous_batch_id={message.previous_batch_id!r} "
                    f"expected={expected_previous!r}",
                    agent_id,
                )
            
            # Store batch information for TelemetryBatch
            self.last_batch_by_agent[agent_id] = message.batch_id
            self.accepted_samples_by_node[message.node_id].extend(
                (message.batch_id, dict(sample)) for sample in message.samples
            )
            self.accepted_samples_by_agent[message.agent_id].extend(
                (message.batch_id, dict(sample)) for sample in message.samples
            )
        
        # Update common state for all message types
        self.last_sequence_by_agent[agent_id] = message.sequence
        self.last_seen_by_node[message.node_id] = now
        self.last_seen_by_agent[agent_id] = now
        self.seen_message_ids.add(message_id)
        self.accepted_message_digests[message_id] = authenticated_digest
        
        return self._ack(message_id, True, "accepted", "message accepted", agent_id)

    def node_health(self) -> dict[str, dict[str, Any]]:
        now = self.clock()
        result: dict[str, dict[str, Any]] = {}
        for node_id in self.expected_nodes:
            last_seen = self.last_seen_by_node.get(node_id)
            age = (now - last_seen).total_seconds() if last_seen is not None else None
            result[node_id] = {
                "last_seen_utc": last_seen.isoformat() if last_seen is not None else None,
                "age_seconds": age,
                "status": (
                    "missing"
                    if last_seen is None
                    else "stale"
                    if age is not None and age > self.stale_after_seconds
                    else "healthy"
                ),
            }
        return result

    def agent_health(self) -> dict[str, dict[str, Any]]:
        """Report rank/GPU-agent loss independently of physical-node health."""
        now = self.clock()
        result: dict[str, dict[str, Any]] = {}
        for agent_id in self.expected_agents:
            last_seen = self.last_seen_by_agent.get(agent_id)
            age = (now - last_seen).total_seconds() if last_seen is not None else None
            result[agent_id] = {
                "node_id": self.registrations[agent_id].node_id,
                "last_seen_utc": last_seen.isoformat() if last_seen is not None else None,
                "age_seconds": age,
                "status": (
                    "missing"
                    if last_seen is None
                    else "stale"
                    if age is not None and age > self.stale_after_seconds
                    else "healthy"
                ),
            }
        return result

    def aggregate_window(self, start_utc: str, end_utc: str) -> dict[str, Any]:
        start = datetime.fromisoformat(start_utc.replace("Z", "+00:00"))
        end = datetime.fromisoformat(end_utc.replace("Z", "+00:00"))
        if end <= start:
            raise ValueError("aggregation window end must be after start")
        health = self.node_health()
        agent_health = self.agent_health()
        nodes: dict[str, Any] = {}
        evidence_ids: set[str] = set()
        for node_id in self.expected_nodes:
            selected = []
            for batch_id, sample in self.accepted_samples_by_node.get(node_id, []):
                observed = datetime.fromisoformat(
                    str(sample["observed_at_utc"]).replace("Z", "+00:00")
                )
                if start <= observed < end:
                    selected.append(sample)
                    evidence_ids.add(batch_id)
            fields: dict[str, float | None] = {}
            for field in TELEMETRY_FIELDS:
                values = [
                    float(sample["fields"][field]["value"])
                    for sample in selected
                    if sample["fields"][field]["supported"]
                    and sample["fields"][field]["value"] is not None
                ]
                fields[field] = statistics.fmean(values) if values else None
            nodes[node_id] = {
                "health": health[node_id],
                "sample_count": len(selected),
                "field_means": fields,
            }
        missing_nodes = [
            node_id
            for node_id, summary in nodes.items()
            if summary["sample_count"] == 0 or summary["health"]["status"] != "healthy"
        ]
        agents: dict[str, Any] = {}
        for agent_id in self.expected_agents:
            selected = []
            for batch_id, sample in self.accepted_samples_by_agent.get(agent_id, []):
                observed = datetime.fromisoformat(
                    str(sample["observed_at_utc"]).replace("Z", "+00:00")
                )
                if start <= observed < end:
                    selected.append(sample)
                    evidence_ids.add(batch_id)
            agents[agent_id] = {
                "node_id": self.registrations[agent_id].node_id,
                "health": agent_health[agent_id],
                "sample_count": len(selected),
                "ranks": sorted({int(sample["rank"]) for sample in selected}),
                "gpu_uuids": sorted({str(sample["gpu_uuid"]) for sample in selected}),
            }
        missing_agents = [
            agent_id
            for agent_id, summary in agents.items()
            if summary["sample_count"] == 0 or summary["health"]["status"] != "healthy"
        ]
        return {
            "window_id": f"window-{uuid.uuid4().hex}",
            "start_utc": start.isoformat(),
            "end_utc": end.isoformat(),
            "nodes": nodes,
            "agents": agents,
            "missing_or_stale_nodes": missing_nodes,
            "missing_or_stale_agents": missing_agents,
            "complete": not missing_nodes and not missing_agents,
            "evidence_batch_ids": sorted(evidence_ids),
            "physical_multi_node_validated": False,
            "prototype_scope": "single_node_dual_gpu",
        }


class DetectorService:
    """Bounded decision API around an injected content-agnostic scoring function."""

    def __init__(
        self,
        scorer: Callable[[dict[str, Any]], float],
        *,
        threshold: float,
        abstention_margin: float,
        model_name: str,
        model_version: str,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if not 0 <= threshold <= 1 or not 0 <= abstention_margin <= 0.5:
            raise ValueError("threshold/margin are outside supported bounds")
        self.scorer = scorer
        self.threshold = threshold
        self.abstention_margin = abstention_margin
        self.model_name = model_name
        self.model_version = model_version
        self.clock = clock or (lambda: datetime.now(timezone.utc))

    def decide(self, aggregation: dict[str, Any]) -> DetectorDecision:
        window_id = aggregation.get("window_id")
        evidence = (str(window_id),) if window_id else ()
        if not aggregation.get("complete"):
            return DetectorDecision(
                decision_id=f"decision-{uuid.uuid4().hex}",
                created_at_utc=self.clock().isoformat(),
                score=None,
                threshold=self.threshold,
                abstained=True,
                reason=f"missing_or_stale_nodes={aggregation.get('missing_or_stale_nodes', [])}",
                model_name=self.model_name,
                model_version=self.model_version,
                evidence_window_ids=evidence,
            )
        score = float(self.scorer(aggregation))
        if not 0 <= score <= 1:
            raise ValueError("detector scorer must return a value in [0, 1]")
        abstained = abs(score - self.threshold) <= self.abstention_margin
        return DetectorDecision(
            decision_id=f"decision-{uuid.uuid4().hex}",
            created_at_utc=self.clock().isoformat(),
            score=score,
            threshold=self.threshold,
            abstained=abstained,
            reason=("score_within_abstention_margin" if abstained else "score_outside_margin"),
            model_name=self.model_name,
            model_version=self.model_version,
            evidence_window_ids=evidence,
        )