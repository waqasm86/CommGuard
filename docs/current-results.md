# Current results and coverage failure analysis

## Bottom line

CommGuard has not established reliable training-versus-inference detection.
The current empirical detector task is effectively **DDP versus idle**, not the
intended broad benign comparison. This analysis is grounded in evidence E4 and
E5 from the [`evidence index`](evidence-index.md).

## What was measured

The historical preflight recorded a strict-ready single-host Kaggle environment
with two Tesla T4 GPUs. The benign pilot then recorded 18 planned, 18 completed,
and zero failed launches at source commit
`1e790895ae3bd0919dde0f383e78fb14f767d359` [E4](evidence-index.md#immutable-local-evidence).
Those completion counts prove process completion only.

The 18 benign run wall durations were approximately 6–12 seconds. With a
two-second warmup, the inference, compute, and host-transfer runs did not leave
enough usable time for even the historical five-second extraction policy. The
current SDK requires at least 35 measured seconds after warmup and treats
30 seconds as the primary window [E4, E5](evidence-index.md#immutable-local-evidence).

## Corpus and derived coverage

| Historical family | Planned | Completed | Feature-valid runs in merged set | Feature rows | Current primary status | Evidence |
|---|---:|---:|---:|---:|---|---|
| DDP training | 3 | 3 | 3 | 5 | fails: only five-second diagnostic rows | E4, E5 |
| independent prefill inference | 3 | 3 | 0 | 0 | missing | E4, E5 |
| synchronized inference | 3 | 3 | 0 | 0 | missing | E4, E5 |
| compute control | 3 | 3 | 0 | 0 | missing | E4, E5 |
| host-transfer control | 3 | 3 | 0 | 0 | missing | E4, E5 |
| benign idle control | 3 | 3 | 3 | included among idle rows | fails: only five-second diagnostic rows | E4, E5 |
| calibration idle (not benign corpus) | not a benign slot | 2 completed | 2 | included among 11 idle rows | leakage under current corpus rules | E5 |

The merged feature artifact therefore has 16 five-second rows from eight runs:
five rows from three DDP runs and 11 rows from five idle runs. It has no
inference, compute, or host-transfer feature row. Two of the idle runs are
calibration controls, and all 26 merged manifests used different historical
environment fingerprints. Neither condition is valid under the current corpus
and true-session contracts [E5](evidence-index.md#immutable-local-evidence).

## Historical split and negative result

The amended analysis used whole-run assignments because the source revision's
environment-fingerprint grouping produced an invalid one-class partition. Its
effective split was four train runs, two validation runs, and two test runs.
The test contained three windows from one DDP run and one idle run
[E5](evidence-index.md#immutable-local-evidence).

| Historical test result | Window-level sample count | Run-level sample count | Balanced accuracy | Training recall | False-negative rate | Evidence |
|---|---:|---:|---:|---:|---:|---|
| PCIe-only selected baseline | 3 | 2 | 0.5 at run level | 0 | 1.0 | E5 |
| Combined diagnostic | 3 | 2 | 1.0 at run level | 1.0 | 0 | E5 |
| Non-PCIe diagnostic | 3 | 2 | 1.0 at run level | 1.0 | 0 | E5 |

The combined and non-PCIe separation is a tiny DDP-versus-idle diagnostic. It
does not support inference-family specificity, cross-session generalization,
or a deployment claim. The historical bootstrap interval is also not accepted:
the current evaluator requires at least 20 independent test runs before it
reports an uncertainty interval. The primary communication-only result is the
negative result: the only training test run was missed.

## What must happen next

The new canonical notebooks must produce three duration-valid 30-second runs
for each of the eight required benign families, using explicit corpus and true
session identities. Coverage is printed and serialized before evaluation. If
any family is missing, detector fitting stops. Only a passing benign evaluation
may unlock the separately approved adversarial plan; the declared final
family/session/configuration holdout remains sealed.
