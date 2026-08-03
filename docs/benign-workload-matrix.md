# Benign workload matrix

CommGuard’s default `standard` profile is a bounded dual-T4 pilot. It contains
three repetitions by default of each required family:

| Family ID | Worker path | Scientific label | Pilot configuration |
|---|---|---|---|
| `ddp_training` | two-rank DDP training | real implementation | AMP, batch 4, sequence 128 |
| `inference_prefill_independent` | independent inference per GPU | real implementation | batch 4, sequence 128 |
| `inference_decode_independent` | independent decode-like inference | controlled proxy | batch 1, context 128 |
| `inference_synchronized` | inference with bounded barriers | controlled proxy | every iteration |
| `control_compute` | independent matrix multiplication | synthetic control | 2048-square matrices |
| `control_host_transfer` | pinned host/device/host copies | synthetic control | 64 MiB payload |
| `control_model_or_checkpoint_load` | CPU state load plus device placement | synthetic control | tiny model state |
| `control_idle` | timed heartbeat control | synthetic control | no GPU operation |

Every non-optional benign configuration declares a five-second warmup and at
least 35 measured seconds. The extra margin protects the 30-second primary
window from asynchronous one-Hz sampler edges. Registry validation also bounds
batch, sequence, hidden size, layers, matrix dimensions, and payload size to
the reviewed pilot envelope. These bounds are planning constraints, not proof
that a particular Kaggle session will avoid OOM or complete successfully.

The opt-in `extended` profile adds one FP32/sequence/model-size DDP variant,
prefill and decode batch/sequence variants, a lower synchronization cadence,
smaller compute and transfer controls, and peer copy. Peer copy is optional and
fails explicitly when CUDA peer access is unavailable; it is not required for
primary coverage.

`run_matrix` creates a versioned `corpora/<corpus-id>-plan.json` before it runs
calibration or a workload. After execution it creates a distinct final corpus
manifest whose allow-list includes completed runs only. Feature extraction then
emits one coverage record per planned slot. The matrix summary reports planned,
completed, failed, primary-feature-valid, and excluded counts by family, along
with stable configuration IDs and exclusion reasons. It reports the primary
coverage-gate decision and deliberately computes no detector metric.

Configuration IDs are explicit research identities such as
`ddp-training-amp-b4-s128-v1`. Repetitions share a configuration ID but have
different plan and run IDs. Session IDs and environment fingerprints are never
used as configuration identities.

No new GPU workload was run while implementing this matrix. Its execution and
performance remain pending compatible dual-T4 evidence.
