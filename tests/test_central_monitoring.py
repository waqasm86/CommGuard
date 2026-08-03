from __future__ import annotations

import copy
import json
from datetime import datetime, timedelta, timezone

import pytest

from commguard.central import (
    AgentRegistration,
    CentralIngestionService,
    DetectorService,
    NodeAgent,
    OfflineFileTransport,
)
from commguard.central.security import sign_message
from commguard.schemas import FIELD_UNITS, TELEMETRY_FIELDS


class FakeClock:
    def __init__(self) -> None:
        self.now = datetime(2026, 8, 2, 9, 0, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


class FakeBackend:
    def __init__(self, clock: FakeClock, offset: float) -> None:
        self.clock = clock
        self.offset = offset
        self.sequence = 0

    def collect(self) -> tuple[dict, ...]:
        samples = []
        for gpu in (0, 1):
            fields = {
                name: {
                    "value": self.offset + self.sequence + gpu + position,
                    "unit": FIELD_UNITS[name],
                    "supported": True,
                    "error": None,
                }
                for position, name in enumerate(TELEMETRY_FIELDS)
            }
            samples.append(
                {
                    "sample_id": f"sample-{self.offset}-{self.sequence}-{gpu}",
                    "observed_at_utc": self.clock().isoformat(),
                    "monotonic_ns": self.sequence * 1_000_000_000,
                    "gpu_index": gpu,
                    "gpu_uuid": f"GPU-{self.offset}-{gpu}",
                    "fields": fields,
                }
            )
        self.sequence += 1
        return tuple(samples)


def setup_two_agents(tmp_path):
    clock = FakeClock()
    registrations = {
        "agent-a": AgentRegistration("agent-a", "node-a", b"secret-a"),
        "agent-b": AgentRegistration("agent-b", "node-b", b"secret-b"),
    }
    service = CentralIngestionService(
        registrations,
        experiment_session_id="session-central",
        expected_nodes=("node-a", "node-b"),
        maximum_clock_skew_seconds=5,
        stale_after_seconds=10,
        clock=clock,
    )
    transport = OfflineFileTransport(tmp_path / "offline", service)
    agents = {
        agent_id: NodeAgent(
            agent_id=agent_id,
            node_id=registration.node_id,
            experiment_session_id="session-central",
            hmac_secret=registration.hmac_secret,
            backend=FakeBackend(clock, float(index * 100)),
            transport=transport,
            clock=clock,
            monotonic_ns=lambda: 1_000_000_000,
        )
        for index, (agent_id, registration) in enumerate(registrations.items())
    }
    return clock, service, agents


def request_payload(root, message_id: str) -> dict:
    return json.loads(
        (root / "offline" / "requests" / f"{message_id}.json").read_text(encoding="utf-8")
    )


def resign(payload: dict, secret: bytes) -> dict:
    payload["signature"] = sign_message(payload, secret)
    return payload


def test_two_agent_offline_ingestion_aggregation_and_decision(tmp_path) -> None:
    clock, service, agents = setup_two_agents(tmp_path)
    batch_ids = []
    for agent in agents.values():
        batch_ids.append(agent.collect_once())
        acknowledgments = agent.flush()
        assert [ack.accepted for ack in acknowledgments] == [True]

    aggregation = service.aggregate_window(
        (clock() - timedelta(seconds=1)).isoformat(),
        (clock() + timedelta(seconds=1)).isoformat(),
    )
    decision = DetectorService(
        lambda window: 0.8,
        threshold=0.5,
        abstention_margin=0.1,
        model_name="test-content-agnostic-scorer",
        model_version="test-only",
        clock=clock,
    ).decide(aggregation)

    assert aggregation["complete"] is True
    assert set(aggregation["nodes"]) == {"node-a", "node-b"}
    assert aggregation["physical_multi_node_validated"] is False
    assert decision.abstained is False
    assert decision.score == 0.8
    assert decision.evidence_window_ids == (aggregation["window_id"],)
    assert len(list((tmp_path / "offline" / "requests").glob("*.json"))) == 2
    assert set(aggregation["evidence_batch_ids"]) == set(batch_ids)
    serialized_requests = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((tmp_path / "offline" / "requests").glob("*.json"))
    ).lower()
    for prohibited in ("prompt", "dataset", "token_text", "model_weight", "secret"):
        assert prohibited not in serialized_requests


def test_idempotent_retry_stale_sequence_invalid_signature_and_chain(tmp_path) -> None:
    _, service, agents = setup_two_agents(tmp_path)
    batch_id = agents["agent-a"].collect_once()
    assert agents["agent-a"].flush()[0].accepted
    original = request_payload(tmp_path, batch_id)

    duplicate = service.ingest(original)
    assert duplicate.accepted is True
    assert duplicate.reason_code == "already_accepted"
    assert len(service.accepted_samples_by_node["node-a"]) == 2

    conflict = copy.deepcopy(original)
    conflict["samples"][0]["fields"]["power_draw_w"]["value"] += 1
    resign(conflict, b"secret-a")
    conflict_ack = service.ingest(conflict)
    assert conflict_ack.accepted is False
    assert conflict_ack.reason_code == "message_id_conflict"
    stale = copy.deepcopy(original)
    stale["batch_id"] = "batch-stale-sequence"
    resign(stale, b"secret-a")
    assert service.ingest(stale).reason_code == "stale_sequence"

    gap = copy.deepcopy(original)
    gap.update({"batch_id": "batch-sequence-gap", "sequence": 2})
    resign(gap, b"secret-a")
    assert service.ingest(gap).reason_code == "sequence_gap"

    invalid = copy.deepcopy(original)
    invalid["batch_id"] = "batch-invalid-signature"
    invalid["sequence"] = 1
    invalid["signature"] = "0" * 64
    assert service.ingest(invalid).reason_code == "invalid_signature"

    broken_chain = copy.deepcopy(original)
    broken_chain.update(
        {
            "batch_id": "batch-broken-chain",
            "sequence": 1,
            "previous_batch_id": "batch-not-the-accepted-parent",
        }
    )
    resign(broken_chain, b"secret-a")
    assert service.ingest(broken_chain).reason_code == "batch_chain_mismatch"


def test_oversize_unknown_protocol_privacy_and_clock_skew_are_rejected(tmp_path) -> None:
    clock, service, agents = setup_two_agents(tmp_path)
    agents["agent-a"].collect_once()
    original = copy.deepcopy(agents["agent-a"].pending[0])

    oversized_service = CentralIngestionService(
        {"agent-a": AgentRegistration("agent-a", "node-a", b"secret-a")},
        experiment_session_id="session-central",
        expected_nodes=("node-a",),
        maximum_payload_bytes=200,
        clock=clock,
    )
    assert oversized_service.ingest(original).reason_code == "payload_too_large"

    unknown = copy.deepcopy(original)
    unknown["protocol_version"] = "999"
    resign(unknown, b"secret-a")
    assert service.ingest(unknown).reason_code == "unsupported_protocol_version"

    private = copy.deepcopy(original)
    private["batch_id"] = "batch-private-field"
    private["samples"][0]["prompt_text"] = "prohibited"
    resign(private, b"secret-a")
    assert service.ingest(private).reason_code == "privacy_violation"

    skewed = copy.deepcopy(original)
    skewed["batch_id"] = "batch-clock-skew"
    skewed["created_at_utc"] = (clock() - timedelta(minutes=5)).isoformat()
    resign(skewed, b"secret-a")
    assert service.ingest(skewed).reason_code == "clock_skew"


def test_node_loss_causes_staleness_and_detector_abstention(tmp_path) -> None:
    clock, service, agents = setup_two_agents(tmp_path)
    for agent in agents.values():
        agent.collect_once()
        assert agent.flush()[0].accepted
    clock.advance(11)
    assert agents["agent-a"].heartbeat().accepted

    aggregation = service.aggregate_window(
        (clock() - timedelta(seconds=20)).isoformat(),
        (clock() + timedelta(seconds=1)).isoformat(),
    )
    decision = DetectorService(
        lambda window: 0.9,
        threshold=0.5,
        abstention_margin=0.1,
        model_name="test",
        model_version="test",
        clock=clock,
    ).decide(aggregation)

    assert service.node_health()["node-a"]["status"] == "healthy"
    assert service.node_health()["node-b"]["status"] == "stale"
    assert aggregation["missing_or_stale_nodes"] == ["node-b"]
    assert decision.abstained is True
    assert decision.score is None


def test_two_rank_agents_on_one_physical_node_and_rank_loss_abstention(tmp_path) -> None:
    clock = FakeClock()
    registrations = {
        "rank-agent-0": AgentRegistration("rank-agent-0", "kaggle-node", b"rank-0"),
        "rank-agent-1": AgentRegistration("rank-agent-1", "kaggle-node", b"rank-1"),
    }
    service = CentralIngestionService(
        registrations,
        experiment_session_id="session-single-node",
        expected_nodes=("kaggle-node",),
        stale_after_seconds=10,
        clock=clock,
    )
    transport = OfflineFileTransport(tmp_path / "single-node", service)

    class RankBackend(FakeBackend):
        def __init__(self, rank: int) -> None:
            super().__init__(clock, float(rank * 100))
            self.rank = rank

        def collect(self) -> tuple[dict, ...]:
            sample = dict(super().collect()[self.rank])
            sample.update({"rank": self.rank, "process_id": 1000 + self.rank})
            return (sample,)

    agents = {
        agent_id: NodeAgent(
            agent_id=agent_id,
            node_id="kaggle-node",
            experiment_session_id="session-single-node",
            hmac_secret=registration.hmac_secret,
            backend=RankBackend(rank),
            transport=transport,
            clock=clock,
            configuration_hash="a" * 64,
        )
        for rank, (agent_id, registration) in enumerate(registrations.items())
    }
    for agent in agents.values():
        agent.collect_once()
        assert agent.flush()[0].accepted
    complete = service.aggregate_window(
        (clock() - timedelta(seconds=1)).isoformat(),
        (clock() + timedelta(seconds=1)).isoformat(),
    )
    assert complete["complete"] is True
    assert set(complete["agents"]) == {"rank-agent-0", "rank-agent-1"}
    assert complete["prototype_scope"] == "single_node_dual_gpu"

    clock.advance(11)
    assert agents["rank-agent-0"].heartbeat().accepted
    incomplete = service.aggregate_window(
        (clock() - timedelta(seconds=20)).isoformat(),
        (clock() + timedelta(seconds=1)).isoformat(),
    )
    assert incomplete["complete"] is False
    assert incomplete["missing_or_stale_agents"] == ["rank-agent-1"]
    decision = DetectorService(
        lambda _: 0.9,
        threshold=0.5,
        abstention_margin=0.1,
        model_name="test",
        model_version="test",
        clock=clock,
    ).decide(incomplete)
    assert decision.abstained is True
    assert decision.score is None


def test_agent_retains_buffer_when_transport_fails() -> None:
    clock = FakeClock()

    service = CentralIngestionService(
        {"agent-a": AgentRegistration("agent-a", "node-a", b"secret-a")},
        experiment_session_id="session-central",
        expected_nodes=("node-a",),
        clock=clock,
    )

    class FlakyTransport:
        def __init__(self):
            self.fail = True

        def send(self, message):
            if self.fail:
                self.fail = False
                raise OSError("temporary offline transport failure")
            return service.ingest(message)

    transport = FlakyTransport()

    agent = NodeAgent(
        agent_id="agent-a",
        node_id="node-a",
        experiment_session_id="session-central",
        hmac_secret=b"secret-a",
        backend=FakeBackend(clock, 0),
        transport=transport,
        clock=clock,
        monotonic_ns=lambda: 1,
    )
    agent.collect_once()

    with pytest.raises(OSError, match="temporary offline"):
        agent.flush()

    assert len(agent.pending) == 1
    assert agent.flush()[0].accepted
    assert agent.pending == []


def test_lost_ack_retry_is_idempotent_and_clears_agent_queue() -> None:
    clock = FakeClock()
    service = CentralIngestionService(
        {"agent-a": AgentRegistration("agent-a", "node-a", b"secret-a")},
        experiment_session_id="session-central",
        expected_nodes=("node-a",),
        clock=clock,
    )

    class AckLosingTransport:
        def __init__(self) -> None:
            self.drop_first_ack = True

        def send(self, message):
            acknowledgment = service.ingest(message)
            if self.drop_first_ack:
                self.drop_first_ack = False
                assert acknowledgment.accepted is True
                raise OSError("acknowledgment lost after server acceptance")
            return acknowledgment

    agent = NodeAgent(
        agent_id="agent-a",
        node_id="node-a",
        experiment_session_id="session-central",
        hmac_secret=b"secret-a",
        backend=FakeBackend(clock, 0),
        transport=AckLosingTransport(),
        clock=clock,
        monotonic_ns=lambda: 1,
    )
    batch_id = agent.collect_once()

    with pytest.raises(OSError, match="acknowledgment lost"):
        agent.flush()
    assert len(agent.pending) == 1
    assert len(service.accepted_samples_by_node["node-a"]) == 2

    acknowledgments = agent.flush()

    assert [(ack.accepted, ack.reason_code) for ack in acknowledgments] == [
        (True, "already_accepted")
    ]
    assert agent.pending == []
    assert agent.last_batch_id == batch_id
    assert len(service.accepted_samples_by_node["node-a"]) == 2
