# CommGuard research report

Status: **historical pilot populated; completion experiment not executed**

> CommGuard is an independent, unofficial research prototype. It is not a
> production monitor, privacy proof, security control, or treaty-verification
> system.

## Abstract

CommGuard asks whether content-agnostic, communication-correlated system
telemetry can distinguish bounded distributed training from benign inference
and controls. Existing dual-T4 evidence is insufficient: the usable historical
task collapsed to DDP versus idle, and the primary PCIe-only baseline missed its
only training test run [E5](../docs/evidence-index.md#immutable-local-evidence).
The broader eight-family, 30-second experiment and all adversarial results are
pending.

## Research question and falsification criteria

Primary question: can 30-second communication-only feature windows distinguish
the declared DDP training family from all required benign inference and control
families while preserving whole-run and true-session grouping?

Falsify or narrow the claim if calibration is unresponsive, required family
coverage is incomplete, classes overlap, whole-run performance collapses,
hard-negative false positives are high, another session does not generalize,
or bounded adversarial families evade a frozen benign-only baseline.

## Environment

Historical evidence records one strict-ready Kaggle host with two Tesla T4 GPUs,
PyTorch 2.10.0+cu128, CUDA 12.8, and NCCL 2.27.5
[E3, E5](../docs/evidence-index.md#immutable-local-evidence). Replace this
paragraph only from a new indexed preflight artifact.

## Corpus coverage

| Family | Planned | Completed | Primary-valid | Exclusion reasons |
|---|---:|---:|---:|---|
| `ddp_training` | pending | pending | pending | pending |
| `inference_prefill_independent` | pending | pending | pending | pending |
| `inference_decode_independent` | pending | pending | pending | pending |
| `inference_synchronized` | pending | pending | pending | pending |
| `control_compute` | pending | pending | pending | pending |
| `control_host_transfer` | pending | pending | pending | pending |
| `control_model_or_checkpoint_load` | pending | pending | pending | pending |
| `control_idle` | pending | pending | pending | pending |

Stop here if the saved primary coverage gate is not passing.

## Split groups

| Split | True session IDs | Run count | Class counts | Required-family coverage |
|---|---|---:|---|---|
| train | pending | pending | pending | pending |
| validation | pending | pending | pending | pending |
| test | pending | pending | pending | pending |

Record the evaluator's actual strategy. Do not infer sessions from environment
fingerprints.

## Primary communication-only result

| Unit | Samples | Balanced accuracy | Precision | Recall | FPR | FNR | Uncertainty |
|---|---:|---:|---:|---:|---:|---:|---|
| window | pending | pending | pending | pending | pending | pending | pending |
| whole run | pending | pending | pending | pending | pending | pending | pending |

## Per-family primary metrics

| Family | Test runs | Test windows | Recall or specificity | FPR | FNR | Notes |
|---|---:|---:|---:|---:|---:|---|
| pending | pending | pending | pending | pending | pending | pending |

## Diagnostic ablations

Non-PCIe, combined, short-window, family-holdout, and configuration-holdout
results belong here and must remain labeled diagnostic.

## Adversarial robustness and costs

| Strategy | Round/status | Runs/windows | Evasion or FPR | Duration ratio | Throughput | Loss proxy | Memory | Sync/byte proxy |
|---|---|---|---:|---:|---:|---:|---:|---|
| `gradient_accumulation` | not executed | — | — | — | — | — | — | — |
| `periodic_local_sgd` | not executed | — | — | — | — | — | — | — |
| `diloco_inspired` | sealed final holdout | — | — | — | — | — | — | — |
| `segmented_runs` | not executed | — | — | — | — | — | — | — |
| `idle_padding` | not executed | — | — | — | — | — | — | — |
| `randomized_synchronization` | not executed | — | — | — | — | — | — | — |
| `mixed_training_inference` | not executed | — | — | — | — | — | — | — |
| `synthetic_communication_decoy` | not executed | — | — | — | — | — | — | — |

Do not score, inspect, or summarize the sealed final holdout before its separate
release approval.

## Negative findings and limitations

The historical effective task was DDP versus idle. Its PCIe-only run-level
balanced accuracy was 0.5, training recall was 0, and false-negative rate was
1.0 on two test runs [E5](../docs/evidence-index.md#immutable-local-evidence).
This is insufficient generalization evidence. Physical multi-node, server-grade
interconnect, privacy, adversarial, and production behavior remain unvalidated.

## Provenance checklist

- Reviewed source commit and clean/remote-visible status: pending
- Input archive name and SHA-256: pending
- Output archive name and SHA-256: pending
- Environment artifact: pending
- Planned/final corpus manifests: pending
- Exact benign and adversarial extraction summaries: pending
- Split and evaluation artifacts: pending
- Evidence-index row: pending

## References

Use [`docs/references.md`](../docs/references.md). Distinguish cited authors'
results from CommGuard's results.
