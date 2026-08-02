# Methodology

The primary unit is a complete run. Windows from a run never cross splits.
Startup is explicitly excluded by default and can be analyzed separately.
Standard profiles use at least three repetitions and randomize bounded run
order. A second Kaggle session is required before cross-session claims.

Primary splitting uses true `experiment_session_id` values and follows a fixed
hierarchy. With at least three suitable sessions, whole sessions become train,
validation, and test only when every partition retains both target classes and
all required families. Otherwise, deterministic class/family-stratified whole
runs are used. Each required family needs at least three runs so it can appear
in all three partitions. Missing or ambiguous legacy session identity is a hard
failure; environment fingerprints are never grouping keys. Family and
configuration holdouts are separate diagnostics and may explicitly have a
one-class test partition, in which case balanced metrics are undefined.

Sampling defaults to 1 Hz. The collector records actual intervals, jitter,
overruns, field missingness, and a conservative sampler duty fraction. Short
controls can compare 1, 2, and 10 Hz; this is not enabled during normal runs.

Calibration records idle-baseline observations and repeated nominal collective
payload groups. It tests per-payload capture rate and repetition count as well
as median-signal rank correlation and dynamic range. Per-payload summaries also
report mean, median absolute deviation, and coefficient of variation. These
thresholds are pragmatic gates, not universal physical laws. Calibration can be
`supported`, `partially_supported`, or `not_supported`; partial support applies
only to the reported reliable payload groups. Calibration failure prevents
standard/extended detector collection unless explicit negative-calibration
research mode is selected.

Feature extraction requires an explicit versioned corpus manifest and produces
one coverage decision for every planned slot. Calibration-designated controls
cannot enter a benign selection. Coverage records expose exact per-GPU row
counts, common overlap, warmup, usable duration, sampling gaps, aligned pairs,
and per-window output counts. A missing or invalid run is never silently
skipped.

The primary window is 30 seconds. Five- and 15-second windows are separately
labeled diagnostic short-window analyses; they are useful for diagnosing old
pilot evidence but never substitute for primary coverage. Primary evaluation
requires every required benign family and its configured minimum run count to
have at least one 30-second window before any model fitting or metric
calculation.

Cross-GPU features use deterministic, no-reuse timestamp pairing within a
configurable tolerance. Independently filtered device arrays are not paired by
position. Per-GPU TX/RX features likewise use values supported in the same
telemetry sample.

Features include distribution summaries, variation, slopes, autocorrelation,
idle/duty fractions, PCIe totals/ratios, missingness, cross-GPU divergence, and
cross-correlation. Evaluation compares majority, simple PCIe rule, logistic
regression, and random forest with PCIe-only, non-PCIe, and combined ablations.
It reports window and run metrics, per-family errors, uncalibrated probability
quality diagnostics, and an abstention region.

The primary result is the 30-second communication-only (currently NVML PCIe)
block with window/run sample counts. Candidate models are fitted on train only,
including median imputation, variance filtering, and logistic scaling. Model
and probability-threshold selection use validation only; test data is evaluated
once and cannot affect those choices. Probabilities are not claimed to be
calibrated. Non-PCIe, combined-feature, short-window, family-holdout, and
configuration-holdout results are labeled diagnostics. Host-transfer and
synchronized-inference false-positive rates are reported explicitly when those
families occur in test.

Run-level uncertainty intervals require at least 20 independent test runs. A
smaller pilot reports an `insufficient independent groups` warning and leaves
the interval null instead of presenting a misleading narrow estimate.

Benign workloads are duration-controlled. Each declares warmup, minimum
steady-state measurement duration, and an optional iteration cap. The current
defaults collect at least 35 measured seconds after a five-second warmup; the
margin prevents asynchronous one-Hz sampler edges from reducing the common
telemetry interval below the 30-second primary window. A supplied cap causes an
explicit failure if duration remains unmet. Rank-local interval events are
validated and their intersection is persisted in the run manifest.

The standard benign pilot has the eight acceptance-criteria family IDs and
three repetitions per family. Its corpus plan is create-only and precedes
calibration and workload execution. A distinct final manifest assigns only
completed runs, after which the coverage summary reports planned, completed,
failed, and primary-feature-valid counts per family. The extended profile is
explicitly opt-in and adds bounded batch, sequence, model-size, precision,
barrier-cadence, transfer, and optional peer-copy variants. See the
[benign workload matrix](benign-workload-matrix.md).

Falsification includes unresponsive counters, substantial class overlap,
grouped-score collapse, duration/startup/framework leakage, hard-negative false
positives, adversarial false negatives, and session instability.

Adversarial evaluation is a later, explicit-approval stage. Its frozen baseline
fits benign training rows only; adversarial rows cannot affect preprocessing,
model selection, or threshold selection. A sealed plan keeps one family plus
declared session/configuration IDs untouched through development and hardening.
Training variants report evasion with duration/throughput/loss/memory/sync and
communication proxies, while the synthetic non-training decoy reports false
positives. See [bounded adversarial research](adversarial-research.md).
