# Next Kaggle experiments

CommGuard is currently a validated dual-GPU monitoring and calibration
prototype. The training/inference corpus and grouped detector evaluation have
not yet been executed on Kaggle, so no training-detection result is claimed.

Run the notebooks in this order on a Kaggle `GPU T4 x2` accelerator:

1. `commguard_calibration_v2.ipynb` measures idle PCIe traffic and repeated
   collective payload groups, then exports the calibration evidence bundle.
2. `commguard_benign_corpus.ipynb` restores that reviewed calibration bundle,
   collects the bounded benign and hard-negative corpus, and exports the
   combined evidence bundle.
3. `commguard_detector_evaluation.ipynb` restores the combined bundle and runs
   grouped feature evaluation only after the required corpus is present.

Each notebook writes an archive under `/kaggle/working`. Download that archive
from the Kaggle Output panel before ending the session. In a later session,
upload it as a Kaggle Dataset, set the notebook's input-bundle path to the file
under `/kaggle/input`, and restore it into the notebook's fresh working
directory. Never delete or replace the read-only `/kaggle/input` source.

The calibration decision can be `supported`, `partially_supported`, or
`not_supported`. A partial result limits subsequent analysis to the explicitly
reported reliable payload groups; it is not evidence that CommGuard detects
training. Do not claim training detection unless the third notebook produces a
non-empty evaluation artifact from complete, grouped runs.
