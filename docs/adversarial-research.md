# Bounded adversarial research

CommGuard’s adversarial workloads are controlled defensive tests of detector
failure modes. They do not alter telemetry, evade provider controls, inspect
model or dataset content, or target a third-party monitoring system. Every
strategy is disabled in the smoke, standard, and extended benign profiles and
requires explicit human approval before preflight or artifact creation.

The dependency-free strategy registry records a stable strategy/family ID,
target label, worker mode, defensive purpose, synchronization semantics,
parameter bounds, and a claim boundary. The bounded pilot strategies are:

| Strategy/family | Defensive question | Bounded default |
|---|---|---|
| `gradient_accumulation` | Does less frequent DDP synchronization reduce detection? | 4 microsteps |
| `periodic_local_sgd` | How does periodic parameter averaging change the signal? | 5 local steps |
| `diloco_inspired` | How does sparse outer synchronization affect a frozen baseline? | 10 inner steps |
| `segmented_runs` | Do short separately launched runs lose primary-window coverage? | 4 × 10 s, 1 s gaps |
| `idle_padding` | What cost accompanies deterministic burst/idle shaping? | 0.5 s padding |
| `randomized_synchronization` | How does seeded synchronization jitter affect detection? | probability 0.25 |
| `mixed_training_inference` | How does alternating forward-only work affect detection? | every second step |
| `synthetic_communication_decoy` | Can bounded non-training collectives cause false positives? | 4 MiB × 4 burst |

The `diloco_inspired` strategy is exactly that: DiLoCo-inspired. It performs
bounded local training and parameter averaging on two GPUs in one host. It
does not implement or claim faithful DiLoCo outer optimization, large-scale
decentralized infrastructure, or the communication reductions reported by the
paper.

## Synchronization and correctness evidence

Sparse schedules are seed-deterministic and rank-independent. Periodic and
DiLoCo-inspired workers average parameter tensors at declared steps and force a
final agreement round when necessary. Completed adversarial manifests require
both ranks to report matching expected and actual synchronization rounds, a
clearly labeled communication-byte proxy, optimizer/inference steps, processed
tokens, throughput, final-loss proxy, wall time, peak memory, and final
parameter agreement for training strategies.

The communication proxy is not a measured NCCL byte count. DDP uses parameter
bytes per gradient-sync round; sparse averaging uses model parameter bytes per
averaging round; the synthetic decoy uses input tensor bytes per all-reduce
call. Run execution records detector score as null with status
`pending_frozen_evaluation`; only a later frozen evaluation may create a score.

Segmented execution is a distinct approval-gated series of torchrun launches,
not an idle period inside one process. It writes a corpus plan before the first
segment and a separate final membership manifest. Individual 10-second
segments must not be concatenated across restart gaps to manufacture a
30-second feature window. Their expected primary exclusion is reported
explicitly.

## Evaluation and untouched holdout

An `AdversarialHoldoutPlan` assigns all eight families to development,
hardening, or one sealed final family and also declares final session and
configuration IDs. Any row matching a final family, session, or configuration
remains `sealed_final_holdout` unless the caller explicitly releases the final
round. The default is sealed.

Primary detector fitting, model selection, threshold selection, and benign
diagnostics use benign rows only. A predeclared random-forest robustness
baseline fits benign training rows only and is then frozen before it scores
approved development/hardening adversarial rows. Training strategies report
evasion/false-negative rate; the non-training decoy reports false-positive
rate. Wall time, throughput, loss, memory, synchronization, and communication
proxies are reported alongside detector scores so apparent evasion is never
presented without its measured cost.

No adversarial GPU workload or detector result was executed while authoring
this implementation. All CUDA behavior, efficiency, quality, evasion, false-
positive, and final-holdout claims remain pending reviewed dual-T4 evidence.
