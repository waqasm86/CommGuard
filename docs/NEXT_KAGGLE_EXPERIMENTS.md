# Next Kaggle experiments

Start with `commguard_calibration_v4.ipynb` in smoke mode, then use a fresh
workspace in full mode. Full calibration plans 30 runs: five idle repetitions
and five repetitions for each 1, 4, 16, 64, and 128 MiB payload. The first
expected archive is `commguard-calibration-prototype-<timestamp>.tar.gz`.

CommGuard has historical dual-T4 calibration and a short benign pilot, but the
derived evidence is DDP versus idle only and does not pass the current primary
coverage policy. See [`current-results.md`](current-results.md). The completion
corpus and grouped detector evaluation have not been executed on Kaggle.

The first calibration-v3 T4 x2 attempt is not an input to this sequence. Its
collective runs executed, but all idle runs failed because worker mode `idle`
was missing; zero idle repetitions were usable and the result was correctly
`not_supported`. Preserve that archive as debugging evidence. Do not give it to
the benign notebook. Start again with a fresh notebook run ID and output
directory after inserting the final pushed commit below.

Run the notebooks in this order on a Kaggle `GPU T4 x2` accelerator:

1. `commguard_calibration_v4.ipynb` measures five idle repetitions and five
   collective repetitions at 1, 4, 16, 64, and 128 MiB for each selected
   standard sampling interval, then exports the calibration evidence bundle
   and its exact artifact hash/reference.
2. `commguard_benign_corpus_v2.ipynb` restores that hash-verified prior-session
   bundle, runs a fresh current-session calibration as the actual gate, records
   both roles without interchanging them, runs a smoke pilot by default, and
   exposes the full 24-run standard corpus as an explicit opt-in. Calibration
   idle runs are never reused as corpus idle runs.
3. `commguard_detector_evaluation_v2.ipynb` restores the combined bundle and runs
   grouped feature evaluation only after the required corpus is present.
4. `commguard_adversarial_redteam_v1.ipynb` requires a passing benign evaluation
   artifact and still leaves every adversarial action disabled until a human
   approves the bounded plan. The final holdout is separately sealed.

Each notebook writes an archive under `/kaggle/working`. Download that archive
from the Kaggle Output panel before ending the session. In a later session,
upload it as a private Kaggle Dataset, set `INPUT_ARCHIVE` to the exact file
under `/kaggle/input`, copy its SHA-256 into `EXPECTED_INPUT_SHA256`, and restore
it into the notebook's fresh working directory. Never delete or replace the
read-only `/kaggle/input` source.

The calibration result can be `supported`, `partially_supported`,
`inconclusive`, `not_supported`, or `failed`. A partial result limits subsequent
analysis to the explicitly reported reliable payload groups; it is not evidence
that CommGuard detects training. Do not claim training detection unless the
third notebook produces a non-empty evaluation artifact after the eight-family,
three-run, 30-second coverage gate.

For the rerun, use a fresh Kaggle copy, insert the final remote-visible
40-character patch commit, select `GPU T4 x2`, and run the calibration notebook
exactly once from its first cell. Share its executed notebook, `.tar.gz`, and
`.sha256` before any benign collection begins.

## Confirmatory calibration-v4 decision

Calibration-v3 at a 0.5-second interval completed operationally but was not
accepted. It produced a `partially_supported` result because the 1 MiB group
passed the repetition-aware capture threshold in only two of five
repetitions. The supported range was 4–128 MiB.

The diagnostic 0.2-second sampling study passed its tested payload groups,
but it did not test the critical 1 MiB group. Therefore, the next scientific
execution is one full calibration-v4 run using:

- 0.2-second sampling;
- five idle repetitions;
- five repetitions at 1, 4, 16, 64, and 128 MiB;
- 30 total observations.

Do not run the benign corpus until all five acceptance fields pass.
