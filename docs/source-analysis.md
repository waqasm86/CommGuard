# Source analysis

The supplied project tree contained 49 files. The analysis used the expanded
version-2 instruction directory as normative, verified every listed SHA-256,
compared both ZIP manifests, parsed the pip snapshot, extracted text/metadata
from every PDF and the DOCX, parsed the project CSV structurally, and inspected
the text/HTML sources. No reference source was modified.

## Instruction bundle

Files `00` through `03` establish defensive scope, a falsification-oriented
single-host question, the fixed Kaggle package snapshot, nine 1 Hz NVML fields,
whole-run grouping, hard negatives, and signal ablations. Files `04` through
`07` require strict architectural separation, public APIs, immutable artifacts,
phased review, and acceptance gates. Files `08` through `10` bound the optional
AWS design, evidence-grounded reporting, non-affiliation, and the prohibition on
AI-written SPAR application responses.

Files `11` through `14` are the controlling multi-GPU additions: one NCCL
process per T4; proof of distinct rank/GPU participation; torchrun isolation;
bounded timeouts and process-tree cleanup; no CPU/Gloo/one-GPU fallback; memory
estimation and no hidden OOM tuning; collective/topology calibration before
classification; host/device, compute, loading, and communication hard
negatives; sampling validation; session holdout; and optional abstention.

The 13 phase prompts restate those requirements as implementation boundaries.
The version-2 ZIP contains the same expanded bundle. The older ZIP lacks the
multi-GPU additions and is superseded. All on-disk version-2 checksums pass.

## Environment snapshot

All three supplied pip-list copies are byte-identical: 935 lines, SHA-256
`f45bb00d807da387d944f692b7626f91737056c735e4681127415ecdccfa0dce`.
The snapshot includes PyTorch 2.10.0+cu128, NCCL 2.27.5, NVML bindings,
transformers/accelerate/datasets/PEFT, NumPy/pandas/SciPy/scikit-learn/PyArrow,
pytest, and optional RAPIDS/CuPy/Dask-CUDA/UCXX components. This supports a
zero-required-dependency core and a Kaggle `--no-deps` install. Snapshot values
are not runtime facts; preflight records active versions.

## Research references

Rahman and Tajdari provide the closest empirical method: nine NVML readings at
1 Hz, temporal windows, grouped runs, broad hard negatives, and iterative
monitor/evader evaluation. Their multi-generation, 100+ GPU-hour corpus and
reported scores cannot be transferred to two T4s. Their results also show why
absolute utilization/power heuristics and random window splits are inadequate.

Seferis and Fist motivate communication/bandwidth as one component of detecting
sequential, disaggregated, and masked compute. Their feasibility argument
depends on cloud-provider workload, identity, inter-node traffic, and compute
records that Kaggle does not expose. CommGuard therefore tests only local
communication-correlated readings and does not implement their governance
system.

Xi et al. show that downscaling can change execution dependencies,
communication, memory, and bottlenecks. PrismLLM's virtual-rank emulation is
relevant future work, but it does not make an ordinary dual-T4 run a faithful
frontier-cluster proxy.

Cankaya's working draft emphasizes separation of evidence capture/commitment
from evaluation, provenance, replay metadata, false-positive attribution, and
layered threat models. Its network taps, active wardens, secure evaluation
environments, physical security, memory challenges, and ZK proofs are far
beyond this SDK.

The two copies of Merzouk et al.'s output-length paper are byte-identical. Its
linear probing methodology concerns hidden-state estimates of remaining token
length and does not support GPU communication classification.

## Project context sources

The structured SPAR CSV has 212 project rows and 19 columns. The William Fowler
row, the extracted project text, and the project-list entry agree on the
technical aim: implement a communication-size training/inference detector,
validate benign workloads, then perform bounded robustness testing and report
failures. The application questions are intentionally excluded from generated
content.

The supplied LessWrong HTML is the "Welcome to LessWrong!" page. It provides
community/rationality context but no technical detector requirement. The SPAR
project list and broader CSV are catalog context; unrelated project rows do not
expand CommGuard's scope.

## Design consequence

The defensible deliverable is a calibration-gated research SDK, not a detector
claim. The SDK must preserve negative calibration, unsupported fields, failures,
and session drift; prevent identity/startup/duration leakage; compare PCIe-only
against non-PCIe and combined signals; and defer every hardware result until a
real Kaggle dual-T4 run produces saved evidence.
