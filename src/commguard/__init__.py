"""CommGuard's stable public API."""

from commguard.corpus import CorpusManifest, PlannedRun
from commguard.environment import preflight
from commguard.environment.preflight import check_environment
from commguard.evaluation import evaluate_detector
from commguard.features import extract_feature_result, extract_features, require_primary_coverage
from commguard.orchestrator import run_experiment, run_matrix
from commguard.provenance import ProvenanceContext
from commguard.reporting import generate_report
from commguard.schemas import load_artifact
from commguard.workloads import list_workloads

__all__ = [
    "check_environment",
    "CorpusManifest",
    "evaluate_detector",
    "extract_feature_result",
    "extract_features",
    "generate_report",
    "list_workloads",
    "load_artifact",
    "PlannedRun",
    "preflight",
    "ProvenanceContext",
    "require_primary_coverage",
    "run_experiment",
    "run_matrix",
]
__version__ = "0.1.0"
