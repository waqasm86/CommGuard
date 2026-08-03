# CommGuard Kaggle prototype research report

Report status: **implemented and locally CPU-tested; not yet executed as the
new full Kaggle protocol; not scientifically accepted**.

> **CommGuard’s Kaggle workflow is a single-node, dual-NVIDIA-T4 research
> prototype. It validates experimental methodology and software behavior on
> two local GPU ranks. It does not establish generalization to two physical
> 8-GPU nodes, NVLink/NVSwitch fabrics, RoCE or InfiniBand networks, large
> frontier-model workloads, or production treaty-verification deployments.**

## 1. Executive summary

CommGuard implements an application-ready research workflow for calibrating
NVML PCIe telemetry against NCCL traffic, collecting duration-valid benign
workloads, evaluating grouped interpretable detectors, and running one bounded
periodic-synchronization red-team study. Code completion is not empirical
validation. The new 30-run calibration, 24-run benign corpus, grouped detector,
and `k = 1, 2, 4, 8, 16` adversarial study remain Kaggle user actions.

## 2. Research question

Can content-agnostic communication measurements entering and leaving GPU ranks
distinguish included training workloads from included inference workloads, and
how does reducing synchronization frequency change that signal and training
utility?

## 3. Relationship to William Fowler's project

The prototype operationalizes the project's proposed monitoring, central
aggregation, benign evaluation, and bounded red-teaming loop. It is independent
and unofficial. It does not answer the SPAR application questions in the
applicant's voice and does not substitute dual-T4 PCIe evidence for the proposed
two-node, 16-GPU experiment.

## 4. Prototype scope

Implemented: one Kaggle host, two Tesla T4 ranks, PyTorch `torchrun`, NCCL, NVML
PCIe telemetry, local rank-agent aggregation, and offline archives. Not
validated: physical nodes, server-grade fabrics, frontier models, production
security, treaty policy, or generalization beyond observed sessions.

## 5. Threat model

The monitor observes content-agnostic rates and health metadata, not prompts,
examples, token text, datasets, model weights, or credentials. A bounded local
research adversary may change synchronization frequency within the controlled
workload. It may not attack Kaggle, external services, other users, or monitoring
infrastructure.

## 6. CommGuard architecture

```text
rank/GPU agent 0 ─┐
                 ├── local collector ── bounded windows ── detector/abstention
rank/GPU agent 1 ─┘
```

The CPU-tested central path validates schemas, signatures, sequence
deduplication, rank loss, bounded aggregation, and incomplete-window abstention.
It is a single-node simulation, not a deployed multi-node service.

## 7. Telemetry fields and privacy considerations

Per sample the current schema stores UTC and monotonic time, target and actual
interval, jitter, rank, PID, GPU index/UUID, PCIe RX/TX, GPU and memory
utilization, power, temperature, clocks, memory used/total, and explicit support
flags. Raw PCIe units are preserved as NVIDIA's documented NVML KB/s label with
the explicit conversion convention `1 NVML KB = 1024 bytes`, the 20 ms query
window, and stored bytes/s. The primary communication rate is RX plus TX bytes/s.

## 8. Calibration methodology

Full mode uses five repetitions of idle and AllReduce payloads 1, 4, 16, 64,
and 128 MiB, with 5 seconds warm-up, at least 30 measured seconds, and about 5
seconds cooldown. Candidate intervals are 1.0, 0.5, and 0.2 seconds; 0.1 seconds
is diagnostic. The default threshold is idle median plus three idle MAD, with a
recorded floor and configurable method. Capture is summarized by repetition and
payload. Launch success and scientific acceptance are separate.

Result table source: `prototype/calibration/calibration_acceptance.json` and
`prototype/calibration/sampling_interval_comparison.csv`.

| Interval | Payload range accepted | Capture rate | Jitter | Overhead | State |
|---:|---|---:|---:|---:|---|
| _not yet executed_ | — | — | — | — | pending |

## 9. Benign workload methodology

The stronger existing eight-family matrix is retained: DDP training,
independent prefill, independent decode, synchronized-inference proxy,
compute control, host-transfer control, model/checkpoint-load control, and idle
control. Three repetitions produce 24 planned runs. Each accepted run requires
both ranks, distinct GPUs, a current supported calibration, and at least one
complete 30-second measurement window. The schedule is deterministic and
resumable by plan/configuration hash.

Completion table source: `prototype/corpus/completion_matrix.csv`.

## 10. Detector methodology

The primary task is training versus inference. Controls are excluded from the
primary binary label and reported separately. Whole runs and sessions are the
grouping units. Supported plans include predefined grouped holdout, independent
session holdout when possible, complete-group GroupKFold and LeaveOneGroupOut,
and deterministic grouped fallbacks. Baselines
are mean-communication threshold, communication-duty threshold, logistic
regression, shallow decision tree, and one random forest diagnostic. Primary
claims use `communication_only`; `auxiliary_only` and
`communication_plus_auxiliary` are secondary.

Results source: `prototype/detector/detector_metrics.json`, `predictions.csv`,
`per_family_metrics.csv`, and `per_session_metrics.csv`.

| Feature set | Model | Balanced accuracy | Training recall | Inference recall | FPR | Coverage |
|---|---|---:|---:|---:|---:|---:|
| _not yet executed_ | — | — | — | — | — | — |

## 11. Adversarial methodology

The first bounded experiment is local-update training with periodic parameter
averaging at `k = 1, 2, 4, 8, 16`. The worker records expected and actual sync
rounds, bounded loss trajectory, final loss, steps/s, tokens/s, pre-final rank
parameter divergence, post-sync agreement, memory peak, and a communication
bytes proxy. The package adds measured communication mean/duty/bursts/peak and
relative communication/utility changes against `k=1`. The frozen benign-only
detector scores each configuration; no
adversarial row is used for fitting or selection. A separate final
family/session/configuration holdout remains sealed.

Results source: `prototype/adversarial/utility_evasion_tradeoff.csv` and
`heldout_evaluation.json`.

| k | Detector score | Decision/abstention | Relative communication | Final loss | Throughput | Utility cost |
|---:|---:|---|---:|---:|---:|---:|
| _not yet executed_ | — | — | — | — | — | — |

## 12. Current historical evidence

Historically executed artifacts remain immutable and indexed in
`docs/evidence-index.md`. The modern calibration-v3 attempt executed collectives
but failed all idle launches because the historical worker lacked idle mode; it
did not pass acceptance. The older short pilot yielded DDP-versus-idle feature
coverage only, and its primary PCIe-only detector missed the held-out training
run. The v4 sampling study is diagnostic, not a canonical gate.

## 13. Results placeholders tied to exact artifacts

- Calibration: `calibration_summary.json`, `calibration_acceptance.json`.
- Corpus: `corpus_acceptance.json`, `completion_matrix.csv`.
- Features: `run_features.csv`, `feature_quality_report.json`.
- Detector: `detector_metrics.csv`, `bootstrap_intervals.json`.
- Red-team: `adversarial_metrics.csv`, `utility_evasion_tradeoff.csv`.
- Integrity: every stage's `provenance.json`, `run_status.json`, and
  `sha256sums.txt`, plus the archive sibling `.sha256`.

Only insert tables after verifying the exact archive and recording its hash.

## 14. Negative and inconclusive findings

Negative, failed, partial, and inconclusive outcomes are retained. Current
historical evidence does not establish reliable training-versus-inference
detection. Small payload failure need not invalidate an explicitly documented
larger-payload sensitivity range, but unsupported or incomplete telemetry causes
abstention rather than an inference label.

## 15. Sampling aliasing and burst-capture limitations

NVML PCIe polling can miss short or phase-aligned communication bursts. Sample
timing has jitter, PCIe counters may be unsupported, and rate samples are not
direct NCCL byte counts. The sampling comparison and optional 0.1-second study
measure this limitation; they do not eliminate it.

## 16. Single-node and dual-T4 limitations

Two local ranks do not reproduce physical node boundaries, network-interface
counters, RDMA, NVLink, NVSwitch, RoCE, InfiniBand, realistic cluster
contention, frontier model scale, or production serving/training software.

## 17. Recommendations for more robust verification

Combine communication measurements with authenticated node health, calibrated
uncertainty, explicit abstention, clock-quality evidence, workload-independent
NIC/RDMA counters, cross-node consistency checks, longer observation windows,
and held-out sessions/hardware. Keep communication-only performance primary so
the incremental benefit of auxiliary telemetry is visible.

## 18. Future two-node, 16-GPU validation plan

Deploy one authenticated agent per physical node; record NIC/RDMA plus
NVLink/NVSwitch counters where available; synchronize clocks; run benign
training, independent and distributed inference, hard controls, and periodic
local-update training across two 8-GPU nodes; group by complete run and session;
seal a final adversarial family; and test transport failure, node loss, counter
support, and out-of-distribution abstention. This is future work, not validated
by Kaggle.

## 19. Reproduction instructions

Follow `docs/KAGGLE_PROTOTYPE_RUNBOOK.md`. Use an exact wheel/archive checksum or
public commit, run calibration smoke then full, verify each archive locally,
run the full resumable corpus, then detector and adversarial notebooks. Preserve
executed notebooks separately. Repeat in a second session for cross-session
validation.

## 20. Evidence index

Historical evidence and supported claim boundaries: `docs/evidence-index.md`.
New artifact status: **not yet executed**. After execution, add archive name,
byte size, SHA-256, source commit/package hash, session ID, acceptance state,
and supported claim before updating any empirical conclusion.

### Status vocabulary

- **Implemented:** source exists.
- **Locally tested:** CPU/static/synthetic checks passed.
- **Kaggle-executable:** designed for strict T4 x2 execution.
- **Historically executed:** immutable earlier evidence exists.
- **Scientifically accepted:** an eligible gate passed; currently not claimed
  for the new workflow.
- **Not yet executed:** new hardware evidence is absent.
- **Future work:** physical multi-node/production validation.
