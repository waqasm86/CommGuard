#!/usr/bin/env python3
"""Generate the canonical, deliberately unexecuted CommGuard notebooks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_ROOT = ROOT / "notebooks"
REPOSITORY_URL = "https://github.com/waqasm86/CommGuard.git"


def markdown(identifier: str, source: str) -> dict[str, object]:
    return {
        "cell_type": "markdown",
        "id": identifier,
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def code(identifier: str, source: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": identifier,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def source_setup(version: str) -> str:
    return f'''from pathlib import Path
import re
import subprocess
import sys

NOTEBOOK_VERSION = "{version}"
REPOSITORY_URL = "{REPOSITORY_URL}"
REVIEWED_COMMIT = ""  # Required: immutable 40-character commit visible on origin.
REPOSITORY = Path("/kaggle/working/commguard-source")

if not re.fullmatch(r"[0-9a-f]{{40}}", REVIEWED_COMMIT):
    raise RuntimeError("Set REVIEWED_COMMIT to the reviewed, pushed 40-character commit SHA.")
if not REPOSITORY.exists():
    subprocess.run(
        ["git", "clone", "--filter=blob:none", "--no-checkout", REPOSITORY_URL, str(REPOSITORY)],
        check=True,
    )
if not (REPOSITORY / ".git").is_dir():
    raise RuntimeError(f"Refusing non-Git source directory: {{REPOSITORY}}")
subprocess.run(["git", "-C", str(REPOSITORY), "fetch", "origin", REVIEWED_COMMIT], check=True)
subprocess.run(
    ["git", "-C", str(REPOSITORY), "checkout", "--detach", REVIEWED_COMMIT], check=True
)
head = subprocess.run(
    ["git", "-C", str(REPOSITORY), "rev-parse", "HEAD"],
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()
dirty = subprocess.run(
    ["git", "-C", str(REPOSITORY), "status", "--porcelain"],
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()
pushed_refs = subprocess.run(
    ["git", "-C", str(REPOSITORY), "branch", "-r", "--contains", head],
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()
if head != REVIEWED_COMMIT or dirty or not pushed_refs:
    raise RuntimeError(
        "Reproducibility gate failed: "
        f"head={{head}} dirty={{bool(dirty)}} pushed={{bool(pushed_refs)}}"
    )
subprocess.run(
    [
        sys.executable, "-m", "pip", "install", "--no-build-isolation", "--no-deps",
        "-e", str(REPOSITORY),
    ],
    check=True,
)
print({{"reviewed_commit": head, "remote_refs": pushed_refs.splitlines()}})
'''


def input_restore(dataset_slug: str, archive_name: str) -> str:
    return f"""from commguard.artifacts import restore_archive, sha256_file

INPUT_ARCHIVE = Path("/kaggle/input/{dataset_slug}/{archive_name}")
EXPECTED_INPUT_SHA256 = ""  # Required: SHA-256 printed by the preceding notebook.
ARTIFACTS = Path("/kaggle/working/commguard-artifacts")

if not re.fullmatch(r"[0-9a-f]{{64}}", EXPECTED_INPUT_SHA256):
    raise RuntimeError("Set EXPECTED_INPUT_SHA256 to the exact 64-character archive hash.")
actual_input_sha256 = sha256_file(INPUT_ARCHIVE)
if actual_input_sha256 != EXPECTED_INPUT_SHA256:
    raise RuntimeError(
        "Input archive hash mismatch: "
        f"expected={{EXPECTED_INPUT_SHA256}} actual={{actual_input_sha256}}"
    )
restore_archive(INPUT_ARCHIVE, ARTIFACTS, expected_sha256=EXPECTED_INPUT_SHA256)
print({{"restored_archive": str(INPUT_ARCHIVE), "sha256": actual_input_sha256}})
"""


def context_cell(version: str, corpus_prefix: str, with_input: bool) -> str:
    input_hash = "EXPECTED_INPUT_SHA256" if with_input else "None"
    return f'''from datetime import datetime, timezone

from commguard.environment.preflight import check_environment, summarize_environment
from commguard.provenance import ProvenanceContext

NOTEBOOK_RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
CONTEXT = ProvenanceContext.create(
    corpus_id=f"corpus-{corpus_prefix}-{{NOTEBOOK_RUN_ID}}",
    experiment_session_id=f"session-{corpus_prefix}-{{NOTEBOOK_RUN_ID}}",
    collection_id=f"collection-{corpus_prefix}-{{NOTEBOOK_RUN_ID}}",
    notebook_version="{version}",
    input_archive_sha256={input_hash},
    random_seed=20260730,
    repository_root=REPOSITORY,
)
if CONTEXT.source_dirty or CONTEXT.source_commit != REVIEWED_COMMIT:
    raise RuntimeError("SDK provenance no longer matches the clean reviewed source commit.")
ENVIRONMENT = check_environment(strict=True, output=ARTIFACTS, provenance=CONTEXT)
print(summarize_environment(ENVIRONMENT))
print({{
    "notebook_run_id": NOTEBOOK_RUN_ID,
    "experiment_session_id": CONTEXT.experiment_session_id,
    "collection_id": CONTEXT.collection_id,
    "corpus_id": CONTEXT.corpus_id,
    "source_commit": CONTEXT.source_commit,
    "input_archive_sha256": CONTEXT.input_archive_sha256,
}})
'''


def export_cell(label: str, next_notebook: str) -> str:
    return f"""from commguard.artifacts import ArtifactStore, sha256_file

ARCHIVE = Path(f"/kaggle/working/{label}-{{NOTEBOOK_RUN_ID}}.tar.gz")
ArtifactStore(ARTIFACTS).export(ARCHIVE)
ARCHIVE_SHA256 = sha256_file(ARCHIVE)
SHA_FILE = ARCHIVE.with_suffix(ARCHIVE.suffix + ".sha256")
SHA_FILE.write_text(f"{{ARCHIVE_SHA256}}  {{ARCHIVE.name}}\\n", encoding="utf-8")
print(f"NEXT STEP: add {{ARCHIVE}} to a private Kaggle dataset without renaming it.")
print(f"NEXT STEP: copy SHA-256 {{ARCHIVE_SHA256}} into EXPECTED_INPUT_SHA256 in {next_notebook}.")
print(
    f"NEXT STEP: set that notebook's REVIEWED_COMMIT to {{REVIEWED_COMMIT}} "
    "and run from the first cell."
)
"""


def calibration_notebook() -> list[dict[str, object]]:
    return [
        markdown(
            "cal-title",
            "# CommGuard calibration v3\n\n"
            "Canonical dual-T4 calibration source. It records environment and bounded collective "
            "observations; it makes no detector claim. Run with Internet enabled only for the "
            "immutable Git fetch, and never add credentials to this notebook.\n",
        ),
        code("cal-source", source_setup("commguard_calibration_v3")),
        code(
            "cal-paths",
            """ARTIFACTS = Path("/kaggle/working/commguard-artifacts")
if ARTIFACTS.exists():
    raise RuntimeError(f"Refusing to overwrite prior artifacts: {ARTIFACTS}")
""",
        ),
        code("cal-context", context_cell("commguard_calibration_v3", "calibration-v3", False)),
        code(
            "cal-run",
            """from commguard.orchestrator import run_calibration_sweep

RUN_STANDARD_CALIBRATION = True
if not RUN_STANDARD_CALIBRATION:
    raise RuntimeError("Enable the bounded standard calibration before export.")
CALIBRATION = run_calibration_sweep(
    output=ARTIFACTS,
    payload_mib=(1, 4, 16, 64),
    repetitions=3,
    timeout_s=180.0,
    provenance=CONTEXT,
)
print({
    "status": CALIBRATION["status"],
    "decision_state": CALIBRATION["decision_state"],
    "idle_usable_repetitions": CALIBRATION["idle_baseline_usable_repetitions"],
    "payload_summaries": CALIBRATION["payload_summaries"],
    "exact_calibration_reference_for_next_notebook": CALIBRATION["reference"],
})
""",
        ),
        markdown(
            "cal-results",
            "## Results\n\n"
            "not executed. The committed notebook contains no runtime result or output.\n",
        ),
        code(
            "cal-export",
            export_cell("commguard-calibration-v3", "commguard_benign_corpus_v2.ipynb"),
        ),
    ]


def benign_notebook() -> list[dict[str, object]]:
    return [
        markdown(
            "benign-title",
            "# CommGuard benign corpus v2\n\n"
            "Restore one exact prior-session calibration as input evidence, then run a fresh "
            "current-session calibration as the actual collection gate. The two artifacts are "
            "never interchangeable. A bounded pilot is the default and the standard 24-run "
            "corpus is an explicit opt-in.\n",
        ),
        code("benign-source", source_setup("commguard_benign_corpus_v2")),
        code(
            "benign-input",
            input_restore("commguard-calibration-v3", "commguard-calibration-v3-REPLACE.tar.gz"),
        ),
        code("benign-context", context_cell("commguard_benign_corpus_v2", "benign-v2", True)),
        code(
            "benign-gate",
            """from commguard.artifacts import sha256_file
from commguard.calibration import build_calibration_reference, verify_calibration_reference

PRIOR_CALIBRATION_ARTIFACT_PATH = Path("results/calibration-REPLACE.json")
EXPECTED_PRIOR_CALIBRATION_SHA256 = ""  # Copy from calibration_v3 output.
if not re.fullmatch(r"[0-9a-f]{64}", EXPECTED_PRIOR_CALIBRATION_SHA256):
    raise RuntimeError("Set the exact prior calibration artifact SHA-256.")
if PRIOR_CALIBRATION_ARTIFACT_PATH.is_absolute() or ".." in PRIOR_CALIBRATION_ARTIFACT_PATH.parts:
    raise RuntimeError("Prior calibration path must be artifact-root-relative.")
prior_path = ARTIFACTS / PRIOR_CALIBRATION_ARTIFACT_PATH
if not prior_path.is_file():
    raise RuntimeError(f"Exact prior calibration is missing: {PRIOR_CALIBRATION_ARTIFACT_PATH}")
if sha256_file(prior_path) != EXPECTED_PRIOR_CALIBRATION_SHA256:
    raise RuntimeError("Exact prior calibration artifact hash does not match.")
PRIOR_CALIBRATION_REFERENCE = build_calibration_reference(
    ARTIFACTS,
    PRIOR_CALIBRATION_ARTIFACT_PATH,
    current_experiment_session_id=CONTEXT.experiment_session_id,
)
if PRIOR_CALIBRATION_REFERENCE["calibration_relationship"] != "prior_session":
    raise RuntimeError("Restored calibration must be labeled prior-session input evidence.")
_, PRIOR_CALIBRATION = verify_calibration_reference(
    ARTIFACTS,
    PRIOR_CALIBRATION_REFERENCE,
    require_current_session=False,
    require_supported=False,
)
print({
    "prior_session_calibration_reference": PRIOR_CALIBRATION_REFERENCE,
    "prior_status_under_its_saved_contract": PRIOR_CALIBRATION["status"],
    "used_as_current_collection_gate": False,
})
""",
        ),
        code(
            "benign-run",
            """from commguard.orchestrator import estimate_matrix, run_matrix

RUN_BENIGN_PILOT = True
RUN_STANDARD_BENIGN_MATRIX = False
RUN_EXPANDED_BENIGN_MATRIX = False

requested = []
if RUN_BENIGN_PILOT:
    requested.append(("smoke", 1))
if RUN_STANDARD_BENIGN_MATRIX:
    requested.append(("standard", 3))
if RUN_EXPANDED_BENIGN_MATRIX:
    requested.append(("extended", 3))
if len(requested) != 1:
    raise RuntimeError("Enable exactly one benign profile per immutable notebook archive.")
PROFILE, REPETITIONS = requested[0]
print({"estimate": estimate_matrix(PROFILE, REPETITIONS), "duration_aware": True})
MATRIX = run_matrix(
    PROFILE,
    output=ARTIFACTS,
    repetitions=REPETITIONS,
    timeout_s=180.0,
    provenance=CONTEXT,
    prior_calibration_reference=PRIOR_CALIBRATION_REFERENCE,
)
CURRENT_CALIBRATION_REFERENCE = MATRIX["calibration_reference"]
if CURRENT_CALIBRATION_REFERENCE["calibration_relationship"] != "current_session":
    raise RuntimeError("Fresh collection calibration was not labeled current_session.")
if CURRENT_CALIBRATION_REFERENCE == PRIOR_CALIBRATION_REFERENCE:
    raise RuntimeError("Prior and current calibration references must not be interchangeable.")
print({
    "prior_session_input_calibration": PRIOR_CALIBRATION_REFERENCE,
    "current_session_gating_calibration": CURRENT_CALIBRATION_REFERENCE,
    "benign_matrix_summary_path_for_next_notebook": MATRIX["summary_artifact"],
    "feature_extraction_summary_for_next_notebook": MATRIX["feature_extraction_summary"],
})
""",
        ),
        code(
            "benign-coverage",
            """COVERAGE_TABLE = [
    {"family": family, **counts}
    for family, counts in sorted(MATRIX["family_counts"].items())
]
for row in COVERAGE_TABLE:
    print(row)
print({
    "primary_coverage_gate": MATRIX["primary_coverage_gate"],
    "detector_metrics_computed": MATRIX["detector_metrics_computed"],
})
""",
        ),
        markdown(
            "benign-results",
            "## Results\n\n"
            "not executed. Coverage and completion are unknown until the notebook is run.\n",
        ),
        code(
            "benign-export",
            export_cell("commguard-benign-corpus-v2", "commguard_detector_evaluation_v2.ipynb"),
        ),
    ]


def detector_notebook() -> list[dict[str, object]]:
    return [
        markdown(
            "detector-title",
            "# CommGuard detector evaluation v2\n\n"
            "Restore an immutable benign archive, print coverage first, and fit only after the "
            "strict eight-family/three-run/30-second gate passes. The communication-only block "
            "is primary; other ablations are diagnostic.\n",
        ),
        code("detector-source", source_setup("commguard_detector_evaluation_v2")),
        code(
            "detector-input",
            input_restore(
                "commguard-benign-corpus-v2",
                "commguard-benign-corpus-v2-REPLACE.tar.gz",
            ),
        ),
        code(
            "detector-context",
            context_cell("commguard_detector_evaluation_v2", "detector-v2", True),
        ),
        code(
            "detector-coverage",
            """import json
from collections import Counter

from commguard.features import (
    PRIMARY_BENIGN_FAMILIES,
    load_extraction_result,
    require_primary_coverage,
)

BENIGN_MATRIX_SUMMARY_PATH = Path("results/matrix-REPLACE.json")
if BENIGN_MATRIX_SUMMARY_PATH.is_absolute() or ".." in BENIGN_MATRIX_SUMMARY_PATH.parts:
    raise RuntimeError("Benign matrix summary path must be artifact-root-relative.")
matrix_summary_path = ARTIFACTS / BENIGN_MATRIX_SUMMARY_PATH
if not matrix_summary_path.is_file():
    raise RuntimeError(f"Exact benign matrix summary is missing: {BENIGN_MATRIX_SUMMARY_PATH}")
BENIGN_MATRIX_SUMMARY = json.loads(matrix_summary_path.read_text(encoding="utf-8"))
BENIGN_EXTRACTION_SUMMARY = Path(BENIGN_MATRIX_SUMMARY["feature_extraction_summary"])
BENIGN_EXTRACTION = load_extraction_result(ARTIFACTS, BENIGN_EXTRACTION_SUMMARY)
if BENIGN_EXTRACTION.calibration_reference != BENIGN_MATRIX_SUMMARY["calibration_reference"]:
    raise RuntimeError("Benign extraction and matrix calibration references differ.")
coverage_by_family = {}
for family in PRIMARY_BENIGN_FAMILIES:
    records = [record for record in BENIGN_EXTRACTION.coverage if record.workload_family == family]
    coverage_by_family[family] = {
        "planned": len(records),
        "included": sum(record.status == "included" for record in records),
        "reason_counts": dict(
            Counter(record.reason_code for record in records if record.reason_code)
        ),
    }
for family, row in sorted(coverage_by_family.items()):
    print({"family": family, **row})
COVERAGE_GATE = require_primary_coverage(
    BENIGN_EXTRACTION,
    required_families=PRIMARY_BENIGN_FAMILIES,
    minimum_runs_per_family=3,
)
print({"coverage_gate": COVERAGE_GATE})
""",
        ),
        code(
            "detector-run",
            """from commguard.evaluation import evaluate_detector

RUN_DETECTOR_EVALUATION = True
if not RUN_DETECTOR_EVALUATION:
    raise RuntimeError("Detector evaluation was disabled after the coverage gate.")
EVALUATION = evaluate_detector(
    input_root=ARTIFACTS,
    output=ARTIFACTS,
    required_families=PRIMARY_BENIGN_FAMILIES,
    minimum_runs_per_family=3,
    benign_extraction_summary=BENIGN_EXTRACTION_SUMMARY.relative_to(ARTIFACTS),
)
print({
    "primary_communication_only": EVALUATION["primary_communication_only"],
    "coverage_gate": EVALUATION["coverage_gate"],
    "warnings": EVALUATION["warnings"],
    "exact_calibration_reference": EVALUATION["calibration_reference"],
    "evaluation_artifact_for_next_notebook": EVALUATION["result_artifact"],
})
""",
        ),
        markdown(
            "detector-results",
            "## Results\n\n"
            "not executed. No accuracy, robustness, or generalization claim is present.\n",
        ),
        code(
            "detector-export",
            export_cell(
                "commguard-detector-evaluation-v2",
                "commguard_adversarial_redteam_v1.ipynb",
            ),
        ),
    ]


def adversarial_notebook() -> list[dict[str, object]]:
    return [
        markdown(
            "adversarial-title",
            "# CommGuard adversarial red-team v1\n\n"
            "Bounded defensive robustness research. Execution is disabled by default and "
            "requires a hash-pinned benign acceptance archive plus explicit approval. The final "
            "family/session/config holdout remains sealed unless separately released.\n",
        ),
        code("adversarial-source", source_setup("commguard_adversarial_redteam_v1")),
        code(
            "adversarial-input",
            input_restore(
                "commguard-detector-evaluation-v2",
                "commguard-detector-evaluation-v2-REPLACE.tar.gz",
            ),
        ),
        code(
            "adversarial-context",
            context_cell("commguard_adversarial_redteam_v1", "adversarial-v1", True),
        ),
        code(
            "adversarial-gate",
            """import json

BENIGN_EVALUATION_ARTIFACT_PATH = Path("results/evaluation-REPLACE.json")
if BENIGN_EVALUATION_ARTIFACT_PATH.is_absolute() or ".." in BENIGN_EVALUATION_ARTIFACT_PATH.parts:
    raise RuntimeError("Benign evaluation path must be artifact-root-relative.")
benign_evaluation_path = ARTIFACTS / BENIGN_EVALUATION_ARTIFACT_PATH
if not benign_evaluation_path.is_file():
    raise RuntimeError("Adversarial work requires the exact saved benign evaluation.")
BENIGN_ACCEPTANCE = json.loads(benign_evaluation_path.read_text(encoding="utf-8"))
if not BENIGN_ACCEPTANCE.get("coverage_gate", {}).get("passed"):
    raise RuntimeError("Adversarial work blocked: benign primary coverage did not pass.")
if not BENIGN_ACCEPTANCE.get("primary_communication_only"):
    raise RuntimeError("Adversarial work blocked: benign primary baseline is missing.")
BENIGN_EXTRACTION_SUMMARY = Path(BENIGN_ACCEPTANCE["feature_source"])
if BENIGN_EXTRACTION_SUMMARY.is_absolute():
    matches = sorted((ARTIFACTS / "features").glob(BENIGN_EXTRACTION_SUMMARY.name))
    if len(matches) != 1:
        raise RuntimeError(
            "Cannot resolve the benign extraction summary inside the restored archive."
        )
    BENIGN_EXTRACTION_SUMMARY = matches[0].relative_to(ARTIFACTS)
print({"accepted_benign_evaluation": str(benign_evaluation_path), "coverage": "passed"})
""",
        ),
        code(
            "adversarial-plan",
            """from commguard.adversarial import ADVERSARIAL_STRATEGIES, AdversarialHoldoutPlan
from commguard.orchestrator import estimate_matrix

HOLDOUT_PLAN = AdversarialHoldoutPlan(
    development_families=(
        "gradient_accumulation",
        "periodic_local_sgd",
        "segmented_runs",
        "idle_padding",
    ),
    hardening_families=(
        "randomized_synchronization",
        "mixed_training_inference",
        "synthetic_communication_decoy",
    ),
    final_family="diloco_inspired",
    final_session_ids=("session-reserved-final-v1",),
    final_config_ids=("diloco-inspired-inner10-v1",),
)
HOLDOUT_PLAN.validate()
for strategy_id, strategy in sorted(ADVERSARIAL_STRATEGIES.items()):
    print({
        "strategy": strategy_id,
        "purpose": strategy.research_purpose,
        "claim_boundary": strategy.claim_boundary,
        "enabled_by_default": strategy.enabled_by_default,
    })
print({"estimate": estimate_matrix("standard", 3), "comparison_only": "benign matrix"})
""",
        ),
        code(
            "adversarial-run",
            """from commguard.orchestrator import run_adversarial_matrix

RUN_ADVERSARIAL_PILOT = False
ADVERSARIAL_HUMAN_APPROVAL = False
RELEASE_FINAL_ADVERSARIAL_HOLDOUT = False
FINAL_HOLDOUT_HUMAN_APPROVAL = False

ADVERSARIAL_MATRIX = None
if RUN_ADVERSARIAL_PILOT:
    if not ADVERSARIAL_HUMAN_APPROVAL:
        raise RuntimeError("Set ADVERSARIAL_HUMAN_APPROVAL only after human review.")
    ADVERSARIAL_MATRIX = run_adversarial_matrix(
        output=ARTIFACTS,
        holdout_plan=HOLDOUT_PLAN,
        adversarial_approval=ADVERSARIAL_HUMAN_APPROVAL,
        release_final_adversarial_holdout=RELEASE_FINAL_ADVERSARIAL_HOLDOUT,
        final_holdout_approval=FINAL_HOLDOUT_HUMAN_APPROVAL,
        repetitions=1,
        timeout_s=180.0,
        provenance=CONTEXT,
    )
    for family, row in sorted(ADVERSARIAL_MATRIX["family_counts"].items()):
        print({"family": family, **row})
else:
    print("Adversarial execution remains disabled; no adversarial artifact was created.")
""",
        ),
        code(
            "adversarial-evaluate",
            """from commguard.evaluation import evaluate_detector

RUN_ADVERSARIAL_EVALUATION = False
ADVERSARIAL_EVALUATION = None
if RUN_ADVERSARIAL_EVALUATION:
    if ADVERSARIAL_MATRIX is None:
        raise RuntimeError("Collect the approved bounded adversarial matrix first.")
    ADVERSARIAL_EVALUATION = evaluate_detector(
        input_root=ARTIFACTS,
        output=ARTIFACTS,
        benign_extraction_summary=BENIGN_EXTRACTION_SUMMARY,
        adversarial_extraction_summary=ADVERSARIAL_MATRIX["feature_extraction_summary"],
        adversarial_holdout_plan=HOLDOUT_PLAN,
        release_final_adversarial_holdout=RELEASE_FINAL_ADVERSARIAL_HOLDOUT,
    )
    print(ADVERSARIAL_EVALUATION["heldout_adversarial_families"])
""",
        ),
        markdown(
            "adversarial-results",
            "## Results\n\n"
            "not executed. No adversarial, evasion, decoy, or final-holdout result is claimed.\n",
        ),
        code(
            "adversarial-export",
            """if ADVERSARIAL_MATRIX is None:
    raise RuntimeError(
        "NEXT STEP: obtain human approval, set RUN_ADVERSARIAL_PILOT and "
        "ADVERSARIAL_HUMAN_APPROVAL to True, then rerun from a fresh Kaggle session."
    )
"""
            + export_cell("commguard-adversarial-redteam-v1", "the evidence index and report"),
        ),
    ]


NOTEBOOKS = {
    "commguard_calibration_v3.ipynb": calibration_notebook,
    "commguard_benign_corpus_v2.ipynb": benign_notebook,
    "commguard_detector_evaluation_v2.ipynb": detector_notebook,
    "commguard_adversarial_redteam_v1.ipynb": adversarial_notebook,
}


def encoded_notebook(cells: list[dict[str, object]]) -> str:
    notebook = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.10"},
            "commguard": {
                "canonical": True,
                "execution_status": "not executed",
                "required_accelerator": "two NVIDIA T4 GPUs",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return json.dumps(notebook, indent=1, ensure_ascii=False) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    mismatches: list[str] = []
    for name, builder in NOTEBOOKS.items():
        expected = encoded_notebook(builder())
        path = NOTEBOOK_ROOT / name
        if args.check:
            if not path.is_file() or path.read_text(encoding="utf-8") != expected:
                mismatches.append(name)
        else:
            path.write_text(expected, encoding="utf-8")
    if mismatches:
        parser.error("generated notebooks differ: " + ", ".join(mismatches))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
