#!/usr/bin/env python3
"""Generate the canonical, deliberately unexecuted CommGuard notebooks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_ROOT = ROOT / "notebooks"
REPOSITORY_URL = "https://github.com/waqasm86/CommGuard.git"
POLICY_VERSION = "commguard-notebook-policy-v3"
SCOPE_DECLARATION = (
    "CommGuard’s Kaggle workflow is a single-node, dual-NVIDIA-T4 research prototype. "
    "It validates experimental methodology and software behavior on two local GPU ranks. "
    "It does not establish generalization to two physical 8-GPU nodes, NVLink/NVSwitch "
    "fabrics, RoCE or InfiniBand networks, large frontier-model workloads, or production "
    "treaty-verification deployments."
)


def markdown(identifier: str, source: str) -> dict[str, object]:
    return {
        "cell_type": "markdown",
        "id": identifier,
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def code(identifier: str, source: str, *, parameters: bool = False) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": identifier,
        "metadata": (
            {"tags": ["parameters"]}
            if parameters or identifier.endswith(("-source", "-input", "-run"))
            else {}
        ),
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def source_setup(version: str) -> str:
    return f'''import hashlib
import importlib
import os
from pathlib import Path
import re
import subprocess
import sys

NOTEBOOK_VERSION = "{version}"
REPOSITORY_URL = "{REPOSITORY_URL}"
INSTALL_SOURCE = "auto"  # auto: wheel, source archive, pinned Git commit, then dev source.
PINNED_PUBLIC_COMMIT = ""  # Required for public Git installation.
EXPECTED_PACKAGE_SHA256 = ""  # Required for a supplied wheel or source archive.
EXPECTED_NOTEBOOK_SHA256 = ""  # SHA-256 of this canonical source notebook.
DEVELOPMENT_SMOKE_TEST = False
DEVELOPMENT_SOURCE = Path("/kaggle/working/commguard-development-source")
REPOSITORY = Path("/kaggle/working/commguard-source")

def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

wheel_candidates = sorted(Path("/kaggle/input").rglob("commguard*.whl"))
archive_candidates = sorted(
    path for path in Path("/kaggle/input").rglob("commguard*")
    if path.is_file() and path.name.endswith((".tar.gz", ".zip"))
    and not any(token in path.name for token in (
        "-prototype-", "review-bundle", "calibration", "benign-corpus",
        "detector-evaluation", "adversarial-redteam",
    ))
)
selected = None
method = INSTALL_SOURCE
if method == "auto":
    method = "wheel" if wheel_candidates else "archive" if archive_candidates else "git"
if method == "wheel":
    if len(wheel_candidates) != 1:
        raise RuntimeError(f"Expected exactly one CommGuard wheel, observed {{wheel_candidates}}")
    selected = wheel_candidates[0]
elif method == "archive":
    if len(archive_candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one CommGuard source archive, observed {{archive_candidates}}"
        )
    selected = archive_candidates[0]

if selected is not None:
    actual_package_sha256 = file_sha256(selected)
    if not re.fullmatch(r"[0-9a-f]{{64}}", EXPECTED_PACKAGE_SHA256):
        raise RuntimeError("Set EXPECTED_PACKAGE_SHA256 for the supplied package.")
    if actual_package_sha256 != EXPECTED_PACKAGE_SHA256:
        raise RuntimeError("Supplied package SHA-256 does not match.")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-deps", str(selected)], check=True
    )
    SOURCE_IDENTITY = f"sha256:{{actual_package_sha256}}"
    SOURCE_DIRTY = False
    INSTALL_PROVENANCE = {{
        "install_source": method,
        "install_path": str(selected),
        "package_sha256": actual_package_sha256,
        "source_identity": SOURCE_IDENTITY,
    }}
elif method == "git":
    if not re.fullmatch(r"[0-9a-f]{{40}}", PINNED_PUBLIC_COMMIT):
        raise RuntimeError("Set PINNED_PUBLIC_COMMIT to a pushed 40-character commit.")
    if not REPOSITORY.exists():
        subprocess.run(
            [
                "git", "clone", "--filter=blob:none", "--no-checkout",
                REPOSITORY_URL, str(REPOSITORY),
            ],
            check=True,
        )
    if not (REPOSITORY / ".git").is_dir():
        raise RuntimeError(f"Refusing non-Git source directory: {{REPOSITORY}}")
    subprocess.run(
        ["git", "-C", str(REPOSITORY), "fetch", "origin", PINNED_PUBLIC_COMMIT],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(REPOSITORY), "checkout", "--detach", PINNED_PUBLIC_COMMIT], check=True
    )
    head = subprocess.run(
        ["git", "-C", str(REPOSITORY), "rev-parse", "HEAD"], check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(REPOSITORY), "status", "--porcelain"], check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    pushed_refs = subprocess.run(
        ["git", "-C", str(REPOSITORY), "branch", "-r", "--contains", head], check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    if head != PINNED_PUBLIC_COMMIT or dirty or not pushed_refs:
        raise RuntimeError("Pinned Git source is dirty, mismatched, or not remote-visible.")
    subprocess.run(
        [
            sys.executable, "-m", "pip", "install", "--no-build-isolation",
            "--no-deps", str(REPOSITORY),
        ],
        check=True,
    )
    SOURCE_IDENTITY = head
    SOURCE_DIRTY = False
    INSTALL_PROVENANCE = {{
        "install_source": "pinned_public_git_commit",
        "repository_url": REPOSITORY_URL,
        "source_identity": head,
        "remote_refs": pushed_refs.splitlines(),
    }}
elif method == "development":
    if not DEVELOPMENT_SMOKE_TEST or not (DEVELOPMENT_SOURCE / "pyproject.toml").is_file():
        raise RuntimeError(
            "Editable development source is allowed only for an explicit smoke test."
        )
    subprocess.run(
        [
            sys.executable, "-m", "pip", "install", "--no-build-isolation",
            "--no-deps", "-e", str(DEVELOPMENT_SOURCE),
        ],
        check=True,
    )
    SOURCE_IDENTITY = "development-editable"
    SOURCE_DIRTY = True
    INSTALL_PROVENANCE = {{
        "install_source": "editable_local_development",
        "source_identity": SOURCE_IDENTITY,
        "development_smoke_only": True,
    }}
else:
    raise RuntimeError(f"Unsupported INSTALL_SOURCE={{method!r}}")

importlib.invalidate_caches()
for module_name in [
    name for name in sys.modules if name == "commguard" or name.startswith("commguard.")
]:
    del sys.modules[module_name]
import commguard
REVIEWED_COMMIT = SOURCE_IDENTITY
PIP_FREEZE = subprocess.run(
    [sys.executable, "-m", "pip", "freeze"], check=True, capture_output=True, text=True
).stdout.splitlines()
INSTALL_PROVENANCE.update({{
    "commguard_version": commguard.__version__,
    "commguard_import": str(Path(commguard.__file__).resolve()),
    "python_version": sys.version,
    "pip_freeze": PIP_FREEZE,
}})
print({{
    "source_identity": SOURCE_IDENTITY,
    "source_dirty": SOURCE_DIRTY,
    "installation": INSTALL_PROVENANCE,
}})
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


def context_cell(
    version: str,
    corpus_prefix: str,
    with_input: bool,
    *,
    run_id_predefined: bool = False,
) -> str:
    input_hash = "EXPECTED_INPUT_SHA256" if with_input else "None"
    run_id_setup = (
        ""
        if run_id_predefined
        else """from datetime import datetime, timezone

NOTEBOOK_RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
"""
    )
    return f'''{run_id_setup}

import socket
from commguard.environment.preflight import check_environment, summarize_environment
from commguard.provenance import ProvenanceContext

NOTEBOOK_FILENAME = f"{{NOTEBOOK_VERSION}}.ipynb"
if not re.fullmatch(r"[0-9a-f]{{64}}", EXPECTED_NOTEBOOK_SHA256):
    raise RuntimeError("Set EXPECTED_NOTEBOOK_SHA256 to the canonical notebook source hash.")
if (REPOSITORY / "notebooks" / NOTEBOOK_FILENAME).is_file():
    actual_notebook_sha256 = file_sha256(REPOSITORY / "notebooks" / NOTEBOOK_FILENAME)
    if actual_notebook_sha256 != EXPECTED_NOTEBOOK_SHA256:
        raise RuntimeError("Canonical notebook SHA-256 does not match the pinned Git source.")
DIRTY_SOURCE_SMOKE_ONLY = bool(SOURCE_DIRTY and DEVELOPMENT_SMOKE_TEST)
if SOURCE_DIRTY and not DIRTY_SOURCE_SMOKE_ONLY:
    raise RuntimeError("Dirty source cannot create accepted research evidence.")
CONTEXT = ProvenanceContext(
    corpus_id=f"corpus-{corpus_prefix}-{{NOTEBOOK_RUN_ID}}",
    experiment_session_id=f"session-{corpus_prefix}-{{NOTEBOOK_RUN_ID}}",
    collection_id=f"collection-{corpus_prefix}-{{NOTEBOOK_RUN_ID}}",
    node_id=socket.gethostname(),
    source_commit=SOURCE_IDENTITY,
    source_dirty=SOURCE_DIRTY,
    notebook_version="{version}",
    input_archive_sha256={input_hash},
    random_seed=20260730,
)
ENVIRONMENT = check_environment(strict=True, output=ARTIFACTS, provenance=CONTEXT)
print(summarize_environment(ENVIRONMENT))
print({{
    "notebook_run_id": NOTEBOOK_RUN_ID,
    "experiment_session_id": CONTEXT.experiment_session_id,
    "collection_id": CONTEXT.collection_id,
    "corpus_id": CONTEXT.corpus_id,
    "source_commit": CONTEXT.source_commit,
    "source_dirty": CONTEXT.source_dirty,
    "notebook_sha256": EXPECTED_NOTEBOOK_SHA256,
    "development_smoke_only": DIRTY_SOURCE_SMOKE_ONLY,
    "input_archive_sha256": CONTEXT.input_archive_sha256,
}})
'''


def export_cell(label: str, next_notebook: str) -> str:
    return f"""from commguard.artifacts import ArtifactStore, sha256_file

ARCHIVE = Path(f"/kaggle/working/{label}-{{NOTEBOOK_RUN_ID}}.tar.gz")
ARCHIVE, SHA_FILE = ArtifactStore(ARTIFACTS).export_with_checksum(ARCHIVE)
ARCHIVE_SHA256 = sha256_file(ARCHIVE)
print(f"NEXT STEP: add {{ARCHIVE}} to a private Kaggle dataset without renaming it.")
print(f"NEXT STEP: copy SHA-256 {{ARCHIVE_SHA256}} into EXPECTED_INPUT_SHA256 in {next_notebook}.")
print(
    "NEXT STEP: set that notebook's install-source parameters and notebook SHA-256, "
    "then run from the first cell."
)
"""


def calibration_notebook() -> list[dict[str, object]]:
    return [
        markdown(
            "cal-title",
            "# CommGuard calibration v3\n\n"
            "Canonical dual-T4 calibration source. It records environment and bounded collective "
            "observations; it makes no detector claim. Run with Internet enabled only for the "
            "immutable Git fetch, and never add credentials to this notebook.\n\n"
            f"> **Prototype scope:** {SCOPE_DECLARATION}\n",
        ),
        code("cal-source", source_setup("commguard_calibration_v3")),
        code(
            "cal-hardware",
            """import shutil

if shutil.which("nvidia-smi") is None:
    raise RuntimeError("nvidia-smi is required for the strict dual-T4 calibration.")
gpu_query = subprocess.run(
    [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,compute_cap",
        "--format=csv,noheader,nounits",
    ],
    check=True,
    capture_output=True,
    text=True,
)
GPU_SUMMARY = []
for line in gpu_query.stdout.splitlines():
    index, name, memory_mib, compute_capability = [part.strip() for part in line.split(",")]
    GPU_SUMMARY.append({
        "index": int(index),
        "name": name,
        "memory_total_mib": int(memory_mib),
        "compute_capability": compute_capability,
    })
if len(GPU_SUMMARY) != 2 or any("T4" not in gpu["name"] for gpu in GPU_SUMMARY):
    raise RuntimeError(f"Expected exactly two Tesla T4 GPUs, observed: {GPU_SUMMARY}")
import torch

if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
    raise RuntimeError("PyTorch must expose exactly two CUDA devices.")
if not torch.distributed.is_nccl_available():
    raise RuntimeError("The reviewed PyTorch build does not expose NCCL.")
print({
    "gpus": GPU_SUMMARY,
    "torch_version": torch.__version__,
    "cuda_version": torch.version.cuda,
    "nccl_available": torch.distributed.is_nccl_available(),
})
""",
        ),
        code(
            "cal-workspace",
            """from datetime import datetime, timezone

NOTEBOOK_RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
ARTIFACTS = Path(f"/kaggle/working/commguard-calibration-v3-{NOTEBOOK_RUN_ID}")
if ARTIFACTS.exists():
    raise RuntimeError(
        f"Refusing reused calibration workspace {ARTIFACTS}; create a new NOTEBOOK_RUN_ID."
    )
ARTIFACTS.mkdir(parents=True, exist_ok=False)
write_probe = ARTIFACTS / ".write-probe"
with write_probe.open("x", encoding="utf-8") as stream:
    stream.write("writable\\n")
write_probe.unlink()
print({"notebook_run_id": NOTEBOOK_RUN_ID, "artifact_workspace": ARTIFACTS.name})
""",
        ),
        code(
            "cal-bootstrap",
            """from commguard.artifacts import ArtifactStore
from commguard.schemas import CURRENT_SCHEMA_VERSION, SCHEMA_VERSION

ArtifactStore(ARTIFACTS).initialize()
BOOTSTRAP = {
    "artifact_kind": "experiment_summary",
    "schema_version": SCHEMA_VERSION,
    "summary_type": "notebook_bootstrap",
    "notebook_version": NOTEBOOK_VERSION,
    "notebook_run_id": NOTEBOOK_RUN_ID,
    "reviewed_commit": REVIEWED_COMMIT,
    "source_repository": "commguard-source",
    "gpu_summary": GPU_SUMMARY,
    "artifact_schema_version": CURRENT_SCHEMA_VERSION,
}
ArtifactStore(ARTIFACTS).write_json("environment/notebook-bootstrap.json", BOOTSTRAP)
print(BOOTSTRAP)
""",
        ),
        code(
            "cal-context",
            context_cell(
                "commguard_calibration_v3",
                "calibration-v3",
                False,
                run_id_predefined=True,
            ),
        ),
        code(
            "cal-run",
            """from commguard.orchestrator import run_calibration_sweep
from commguard.telemetry import compare_sampling_intervals

RUN_MODE = "smoke"  # "smoke" or "full"
if RUN_MODE not in {"smoke", "full"}:
    raise RuntimeError("RUN_MODE must be smoke or full.")
DEVELOPMENT_SMOKE_ONLY = RUN_MODE == "smoke"
if SOURCE_DIRTY and RUN_MODE != "smoke":
    raise RuntimeError("Dirty editable source is restricted to RUN_MODE='smoke'.")
DEVELOPMENT_SMOKE_ONLY = DEVELOPMENT_SMOKE_ONLY or DIRTY_SOURCE_SMOKE_ONLY
PAYLOADS = (16,) if DEVELOPMENT_SMOKE_ONLY else (1, 4, 16, 64, 128)
REPETITIONS = 1 if DEVELOPMENT_SMOKE_ONLY else 5
SAMPLING_INTERVAL_S = 0.5
SAMPLING_COMPARISON = compare_sampling_intervals(
    f"sampling-{NOTEBOOK_RUN_ID}",
    intervals_s=(1.0, 0.5, 0.2),
    duration_s=3.0 if DEVELOPMENT_SMOKE_ONLY else 10.0,
)
print({
    "run_mode": RUN_MODE,
    "development_smoke_only": DEVELOPMENT_SMOKE_ONLY,
    "scientific_acceptance_eligible": not DEVELOPMENT_SMOKE_ONLY,
    "idle_repetitions": REPETITIONS,
    "collective": "all_reduce",
    "payload_mib": list(PAYLOADS),
    "total_runs": REPETITIONS * (1 + len(PAYLOADS)),
})
CALIBRATION = run_calibration_sweep(
    output=ARTIFACTS,
    payload_mib=PAYLOADS,
    repetitions=REPETITIONS,
    sampling_interval_s=SAMPLING_INTERVAL_S,
    timeout_s=180.0,
    provenance=CONTEXT,
    progress_callback=lambda event: print({"calibration_progress": event}),
    development_smoke_only=DEVELOPMENT_SMOKE_ONLY,
)
print({
    "status": CALIBRATION["status"],
    "result_state": CALIBRATION["result_state"],
    "decision_state": CALIBRATION["decision_state"],
    "idle_usable_repetitions": CALIBRATION["idle_baseline_usable_repetitions"],
    "payload_summaries": CALIBRATION["payload_summaries"],
    "standard_sweep_validation": CALIBRATION["standard_sweep_validation"],
    "exact_calibration_reference_for_next_notebook": CALIBRATION["reference"],
})
""",
        ),
        code(
            "cal-package",
            """from commguard.artifacts import materialize_calibration_package

CALIBRATION_PACKAGE = materialize_calibration_package(
    ARTIFACTS,
    CALIBRATION,
    SAMPLING_COMPARISON,
    environment=ENVIRONMENT,
    provenance={**CONTEXT.to_dict(), **INSTALL_PROVENANCE},
    notebook_filename=NOTEBOOK_FILENAME,
    notebook_sha256=EXPECTED_NOTEBOOK_SHA256,
    configuration={
        "run_mode": RUN_MODE,
        "payload_mib": list(PAYLOADS),
        "repetitions": REPETITIONS,
        "sampling_interval_s": SAMPLING_INTERVAL_S,
    },
    development_smoke_only=DEVELOPMENT_SMOKE_ONLY,
)
print({"machine_readable_package": str(CALIBRATION_PACKAGE)})
""",
        ),
        markdown(
            "cal-results",
            "## Results\n\n"
            "not executed. The committed notebook contains no runtime result or output.\n",
        ),
        code(
            "cal-export",
            """from commguard.artifacts import ArtifactStore, sha256_file

ARCHIVE = Path(f"/kaggle/working/commguard-calibration-prototype-{NOTEBOOK_RUN_ID}.tar.gz")
ARCHIVE, SHA_FILE = ArtifactStore(ARTIFACTS).export_with_checksum(ARCHIVE)
ARCHIVE_SHA256 = sha256_file(ARCHIVE)
print({
    "archive": str(ARCHIVE),
    "archive_sha256": ARCHIVE_SHA256,
    "sha256_file": str(SHA_FILE),
    "calibration_status": CALIBRATION["status"],
    "calibration_artifact_reference": CALIBRATION["reference"],
})
if DEVELOPMENT_SMOKE_ONLY:
    print("DEVELOPMENT SMOKE ONLY: run full mode in a new workspace before benign collection.")
elif CALIBRATION["result_state"] != "supported" or not CALIBRATION["modern_capture_gate_passed"]:
    print("FAILED/INCONCLUSIVE EVIDENCE WAS PRESERVED. Do not run the benign notebook.")
    raise RuntimeError(
        "Modern idle-aware calibration is not supported; download the diagnostic archive "
        "and SHA-256 file, then investigate before creating a fresh NOTEBOOK_RUN_ID."
    )
print(f"NEXT STEP: add {ARCHIVE} to a private Kaggle dataset without renaming it.")
print(
    "NEXT STEP: copy the exact archive SHA-256 and calibration artifact reference into "
    "commguard_benign_corpus_v2.ipynb."
)
""",
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
            "corpus is an explicit opt-in.\n\n"
            f"> **Prototype scope:** {SCOPE_DECLARATION}\n",
        ),
        code("benign-source", source_setup("commguard_benign_corpus_v2")),
        code(
            "benign-input",
            input_restore(
                "commguard-calibration-prototype",
                "commguard-calibration-prototype-REPLACE.tar.gz",
            ),
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

RUN_MODE = "smoke"  # "smoke", "full", or "expanded"
RUN_CONFIGURATIONS = {
    "smoke": ("smoke", 1),
    "full": ("standard", 3),  # Eight families × three repetitions = 24 runs.
    "expanded": ("extended", 3),
}
if RUN_MODE not in RUN_CONFIGURATIONS:
    raise RuntimeError("RUN_MODE must be smoke, full, or expanded.")
PROFILE, REPETITIONS = RUN_CONFIGURATIONS[RUN_MODE]
DEVELOPMENT_SMOKE_ONLY = RUN_MODE == "smoke"
if SOURCE_DIRTY and RUN_MODE != "smoke":
    raise RuntimeError("Dirty editable source is restricted to RUN_MODE='smoke'.")
DEVELOPMENT_SMOKE_ONLY = DEVELOPMENT_SMOKE_ONLY or DIRTY_SOURCE_SMOKE_ONLY
print({"estimate": estimate_matrix(PROFILE, REPETITIONS), "duration_aware": True})
MATRIX = run_matrix(
    PROFILE,
    output=ARTIFACTS,
    repetitions=REPETITIONS,
    timeout_s=180.0,
    provenance=CONTEXT,
    prior_calibration_reference=PRIOR_CALIBRATION_REFERENCE,
    resume=True,
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
            "benign-package",
            """from commguard.artifacts import materialize_corpus_package

CORPUS_PACKAGE = materialize_corpus_package(
    ARTIFACTS,
    MATRIX,
    environment=ENVIRONMENT,
    provenance={**CONTEXT.to_dict(), **INSTALL_PROVENANCE},
    notebook_filename=NOTEBOOK_FILENAME,
    notebook_sha256=EXPECTED_NOTEBOOK_SHA256,
    configuration={"run_mode": RUN_MODE, "profile": PROFILE, "repetitions": REPETITIONS},
    development_smoke_only=DEVELOPMENT_SMOKE_ONLY,
)
print({"machine_readable_package": str(CORPUS_PACKAGE)})
""",
        ),
        code(
            "benign-export",
            export_cell(
                "commguard-benign-corpus-prototype",
                "commguard_detector_evaluation_v2.ipynb",
            ),
        ),
    ]


def detector_notebook() -> list[dict[str, object]]:
    return [
        markdown(
            "detector-title",
            "# CommGuard detector evaluation v2\n\n"
            "Restore an immutable benign archive, print coverage first, and fit only after the "
            "strict eight-family/three-run/30-second gate passes. The communication-only block "
            "is primary; other ablations are diagnostic.\n\n"
            f"> **Prototype scope:** {SCOPE_DECLARATION}\n",
        ),
        code("detector-source", source_setup("commguard_detector_evaluation_v2")),
        code(
            "detector-input",
            input_restore(
                "commguard-benign-corpus-prototype",
                "commguard-benign-corpus-prototype-REPLACE.tar.gz",
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

RUN_MODE = "smoke"  # "smoke" or "full"
if RUN_MODE not in {"smoke", "full"}:
    raise RuntimeError("RUN_MODE must be smoke or full.")
if SOURCE_DIRTY and RUN_MODE != "smoke":
    raise RuntimeError("Dirty editable source is restricted to RUN_MODE='smoke'.")
DEVELOPMENT_SMOKE_ONLY = RUN_MODE == "smoke" or DIRTY_SOURCE_SMOKE_ONLY
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
        code(
            "detector-package",
            """from commguard.artifacts import materialize_detector_package

DETECTOR_PACKAGE = materialize_detector_package(
    ARTIFACTS,
    EVALUATION,
    environment=ENVIRONMENT,
    provenance={**CONTEXT.to_dict(), **INSTALL_PROVENANCE},
    notebook_filename=NOTEBOOK_FILENAME,
    notebook_sha256=EXPECTED_NOTEBOOK_SHA256,
    configuration={
        "task": "training_vs_inference",
        "minimum_runs_per_family": 3,
        "primary_feature_set": "communication_only",
        "run_mode": RUN_MODE,
        "development_smoke_only": DEVELOPMENT_SMOKE_ONLY,
    },
)
print({"machine_readable_package": str(DETECTOR_PACKAGE)})
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
                "commguard-detector-evaluation-prototype",
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
            "family/session/config holdout remains sealed unless separately released.\n\n"
            f"> **Prototype scope:** {SCOPE_DECLARATION}\n",
        ),
        code("adversarial-source", source_setup("commguard_adversarial_redteam_v1")),
        code(
            "adversarial-input",
            input_restore(
                "commguard-detector-evaluation-prototype",
                "commguard-detector-evaluation-prototype-REPLACE.tar.gz",
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
            """from commguard.orchestrator import run_periodic_synchronization_study

RUN_MODE = "smoke"  # "smoke" or "full"
if RUN_MODE not in {"smoke", "full"}:
    raise RuntimeError("RUN_MODE must be smoke or full.")
if SOURCE_DIRTY and RUN_MODE != "smoke":
    raise RuntimeError("Dirty editable source is restricted to RUN_MODE='smoke'.")
DEVELOPMENT_SMOKE_ONLY = RUN_MODE == "smoke" or DIRTY_SOURCE_SMOKE_ONLY
RUN_FULL_PERIODIC_SYNCHRONIZATION_STUDY = RUN_MODE == "full"
ADVERSARIAL_HUMAN_APPROVAL = False
SYNCHRONIZATION_INTERVALS = (1, 2, 4, 8, 16)

ADVERSARIAL_MATRIX = None
if RUN_FULL_PERIODIC_SYNCHRONIZATION_STUDY:
    if not ADVERSARIAL_HUMAN_APPROVAL:
        raise RuntimeError("Set ADVERSARIAL_HUMAN_APPROVAL only after human review.")
    ADVERSARIAL_MATRIX = run_periodic_synchronization_study(
        output=ARTIFACTS,
        holdout_plan=HOLDOUT_PLAN,
        adversarial_approval=ADVERSARIAL_HUMAN_APPROVAL,
        synchronization_intervals=SYNCHRONIZATION_INTERVALS,
        repetitions=1,
        timeout_s=180.0,
        provenance=CONTEXT,
    )
    for family, row in sorted(ADVERSARIAL_MATRIX["family_counts"].items()):
        print({"family": family, **row})
else:
    from commguard.artifacts import ArtifactStore
    from commguard.scope import with_prototype_scope

    ArtifactStore(ARTIFACTS).write_json(
        "prototype/adversarial/run_status.json",
        with_prototype_scope({
            "execution_state": "development_smoke_plan_only",
            "development_smoke_only": True,
            "scientific_acceptance_eligible": False,
            "bounded_research_redteam": True,
            "reason": "Full periodic synchronization execution was not enabled.",
        }),
        validate=False,
    )
    print("Adversarial smoke mode recorded a plan-only run status; no attack was executed.")
""",
        ),
        code(
            "adversarial-evaluate",
            """from commguard.evaluation import evaluate_detector

RUN_ADVERSARIAL_EVALUATION = RUN_FULL_PERIODIC_SYNCHRONIZATION_STUDY
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
        release_final_adversarial_holdout=False,
    )
    print(ADVERSARIAL_EVALUATION["heldout_adversarial_families"])
""",
        ),
        code(
            "adversarial-package",
            """from commguard.artifacts import materialize_adversarial_package

if DEVELOPMENT_SMOKE_ONLY:
    ADVERSARIAL_PACKAGE = ARTIFACTS / "prototype/adversarial"
elif ADVERSARIAL_MATRIX is None or ADVERSARIAL_EVALUATION is None:
    raise RuntimeError("Run and evaluate the approved periodic synchronization study first.")
else:
    ADVERSARIAL_PACKAGE = materialize_adversarial_package(
        ARTIFACTS,
        ADVERSARIAL_MATRIX,
        ADVERSARIAL_EVALUATION,
        environment=ENVIRONMENT,
        provenance={**CONTEXT.to_dict(), **INSTALL_PROVENANCE},
        notebook_filename=NOTEBOOK_FILENAME,
        notebook_sha256=EXPECTED_NOTEBOOK_SHA256,
        configuration={
            "strategy": "periodic_synchronization_local_update_training",
            "synchronization_intervals": list(SYNCHRONIZATION_INTERVALS),
            "detector_frozen_before_adversarial_scoring": True,
        },
    )
print({"machine_readable_package": str(ADVERSARIAL_PACKAGE)})
""",
        ),
        markdown(
            "adversarial-results",
            "## Results\n\n"
            "not executed. No adversarial, evasion, decoy, or final-holdout result is claimed.\n",
        ),
        code(
            "adversarial-export",
            """if ADVERSARIAL_MATRIX is None and not DEVELOPMENT_SMOKE_ONLY:
    raise RuntimeError(
        "NEXT STEP: obtain human approval, set RUN_MODE='full' and "
        "ADVERSARIAL_HUMAN_APPROVAL=True, then rerun from a fresh Kaggle session."
    )
"""
            + export_cell(
                "commguard-adversarial-redteam-prototype",
                "the evidence index and report",
            ),
        ),
    ]


NOTEBOOKS = {
    "commguard_calibration_v3.ipynb": calibration_notebook,
    "commguard_benign_corpus_v2.ipynb": benign_notebook,
    "commguard_detector_evaluation_v2.ipynb": detector_notebook,
    "commguard_adversarial_redteam_v1.ipynb": adversarial_notebook,
}

NOTEBOOK_ROLES = {
    "commguard_calibration_v3.ipynb": (
        "calibrate dual-T4 telemetry support and accepted payload sensitivity"
    ),
    "commguard_benign_corpus_v2.ipynb": (
        "collect a duration-valid, resumable benign workload corpus"
    ),
    "commguard_detector_evaluation_v2.ipynb": (
        "evaluate grouped communication-only and auxiliary detector baselines"
    ),
    "commguard_adversarial_redteam_v1.ipynb": (
        "evaluate bounded periodic-synchronization training with a sealed holdout"
    ),
}


def canonical_inventory() -> dict[str, object]:
    """Return the ordered machine-readable notebook policy inventory."""
    return {
        "schema_version": 2,
        "policy_version": POLICY_VERSION,
        "canonical_notebooks": [
            {"order": order, "filename": name, "role": NOTEBOOK_ROLES[name]}
            for order, name in enumerate(NOTEBOOKS, start=1)
        ],
        "diagnostic_notebooks": [
            {
                "filename": "diagnostics/commguard_calibration_v4_sampling_study.ipynb",
                "role": (
                    "diagnostic sampling-resolution study; never a canonical calibration gate"
                ),
            }
        ],
        "executed_name_pattern": "commguard-*.ipynb",
        "policy_document": "../docs/notebook-policy.md",
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
            "language_info": {"name": "python", "version": "3.11"},
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
    inventory_path = NOTEBOOK_ROOT / "canonical_notebooks.json"
    expected_inventory = json.dumps(canonical_inventory(), indent=2, ensure_ascii=False) + "\n"
    if args.check:
        if (
            not inventory_path.is_file()
            or inventory_path.read_text(encoding="utf-8") != expected_inventory
        ):
            mismatches.append(inventory_path.name)
    else:
        inventory_path.write_text(expected_inventory, encoding="utf-8")
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
