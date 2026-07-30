# Methodology

The primary unit is a complete run. Windows from a run never cross splits.
Startup is explicitly excluded by default and can be analyzed separately.
Standard profiles use at least three repetitions and randomize bounded run
order. A second Kaggle session is required before cross-session claims.

Sampling defaults to 1 Hz. The collector records actual intervals, jitter,
overruns, field missingness, and a conservative sampler duty fraction. Short
controls can compare 1, 2, and 10 Hz; this is not enabled during normal runs.

Calibration sweeps nominal collective payloads and tests rank correlation plus
signal dynamic range. These thresholds are pragmatic gates, not universal
physical laws. Calibration failure prevents standard/extended detector
collection unless explicit negative-calibration research mode is selected.

Features include distribution summaries, variation, slopes, autocorrelation,
idle/duty fractions, PCIe totals/ratios, missingness, cross-GPU divergence, and
cross-correlation. Evaluation compares majority, simple PCIe rule, logistic
regression, and random forest with PCIe-only, non-PCIe, and combined ablations.
It reports window and run metrics, per-family errors, calibration quality, and
an abstention region.

Falsification includes unresponsive counters, substantial class overlap,
grouped-score collapse, duration/startup/framework leakage, hard-negative false
positives, adversarial false negatives, and session instability.
