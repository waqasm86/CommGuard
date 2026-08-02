# Next Kaggle experiments

CommGuard has historical dual-T4 calibration and a short benign pilot, but the
derived evidence is DDP versus idle only and does not pass the current primary
coverage policy. See [`current-results.md`](current-results.md). The completion
corpus and grouped detector evaluation have not been executed on Kaggle.

Run the notebooks in this order on a Kaggle `GPU T4 x2` accelerator:

1. `commguard_calibration_v3.ipynb` measures repeated
   collective payload groups, then exports the calibration evidence bundle.
2. `commguard_benign_corpus_v2.ipynb` restores that hash-verified calibration
   bundle, runs a smoke pilot by default, and exposes the full 24-run standard
   corpus as an explicit opt-in.
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

The calibration decision can be `supported`, `partially_supported`, or
`not_supported`. A partial result limits subsequent analysis to the explicitly
reported reliable payload groups; it is not evidence that CommGuard detects
training. Do not claim training detection unless the third notebook produces a
non-empty evaluation artifact after the eight-family, three-run, 30-second
coverage gate.
