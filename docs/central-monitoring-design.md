# Central monitoring reference design

The Kaggle reference path models two rank/GPU agents on the **same physical
node**. Agent sequence state and health are independent, so loss or staleness of
either rank makes a window incomplete and forces detector abstention. Raw
events and aggregate windows remain local. This is not evidence of a physical
two-node deployment.

Future deployment requires one agent per physical node, authenticated secure
transport, key lifecycle, clock synchronization, persistent storage, NIC/RDMA
counters, NVLink/NVSwitch counters, RoCE/InfiniBand monitoring, and central
policy enforcement.

## Status

This central server is a CPU-tested local reference implementation. It has not been deployed or
validated across physical GPU nodes. No current result supports a multi-node,
server-grade, latency, availability, or detector-performance claim.

## Components and flow

Each node-local `NodeAgent` reads a pluggable backend, converts readings to the
strict content-agnostic sample allow-list, buffers a versioned batch, and signs
the canonical JSON with its per-agent HMAC secret. `OfflineFileTransport` is the
reproducible default: it writes create-only request and acknowledgment files and
invokes the central service in process. A bounded agent loop and retained
pending queue make transport failures retryable without dropping the batch.

`CentralIngestionService` checks payload size before parsing, then protocol
version, registered agent/node binding, HMAC, experiment session, exact schema,
privacy allow-list, unique message ID, strictly increasing sequence, clock skew,
and previous-batch linkage. Only accepted telemetry enters in-memory reference
state. An exact authenticated retry of an already accepted envelope receives an
idempotent `already_accepted` acknowledgment and does not ingest samples again;
reuse of that message ID with different authenticated content is rejected as a
conflict. Heartbeats update node liveness without adding telemetry.

Aggregation selects samples by UTC window because monotonic clocks are only
node-local. It reports per-node sample counts, field means, health, missing or
stale nodes, accepted batch IDs, and `physical_multi_node_validated: false`.
The `DetectorService` wraps an injected scorer and returns score, threshold,
abstention/reason, model name/version, and the evidence window ID. Missing or
stale nodes force abstention before the scorer is called.

## Security and privacy controls

- HMAC-SHA256 uses canonical JSON and a distinct configured secret per agent.
- Accepted sequences are strictly increasing. Exact acknowledgment-loss retries
  are idempotent; changed content under an accepted ID is rejected.
- Batch chaining detects gaps or out-of-order parents.
- UTC message/sample skew, node staleness, and payload size are bounded.
- Only GPU/sample identity, timing, and the nine declared telemetry readings are
  accepted. Content/model/data/credential key fragments are rejected.
- Standard request logging is disabled by the optional HTTP handler so
  signatures and credentials are not accidentally logged.

These controls are a reference baseline, not a production security review.
HMAC secrets must come from an external secret manager; they must never be
committed or placed in evidence artifacts.

## Optional online deployment

`HttpTransport` requires an `https://` endpoint. `build_http_handler` supplies a
minimal standard-library POST handler for `/v1/ingest`; the operator owns the
server lifecycle and must wrap it in verified TLS. A real deployment should use
TLS 1.3 where available, mutual TLS or equivalent workload identity, external
secret rotation, rate limiting, durable replay/sequence state, bounded queues,
audited redacted logging, time synchronization monitoring, and independent
penetration/reliability testing. Running the reference handler without these
controls is not a supported production configuration.

## Local evidence boundary

CPU tests simulate two agents, aggregation, a decision, replay, stale sequence,
acknowledgment loss with exact retry, conflicting ID reuse, bad HMAC, bad batch
chaining, oversized payload, unknown protocol, prohibited fields, clock skew,
buffered transport failure, and node loss. They do not exercise sockets, TLS,
multiple hosts, GPUs, or real detector models.

Before a physical pilot, an operator must specify node/workload identity,
secret provisioning and rotation, TLS termination and peer verification,
durable replay/sequence state, queue backpressure, synchronized-clock alerting,
retention/deletion policy, role-based evidence access, GPU-to-NIC attribution,
failure recovery, and an independent security/reliability review. These are
deployment prerequisites, not features validated by the local simulation.
