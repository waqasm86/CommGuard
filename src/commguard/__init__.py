"""CommGuard's stable public API."""

from commguard.environment import preflight
from commguard.environment.preflight import check_environment
from commguard.evaluation import evaluate_detector
from commguard.features import extract_features
from commguard.orchestrator import run_experiment, run_matrix
from commguard.reporting import generate_report
from commguard.schemas import load_artifact
from commguard.workloads import list_workloads

__all__ = [
    "check_environment",
    "evaluate_detector",
    "extract_features",
    "generate_report",
    "list_workloads",
    "load_artifact",
    "preflight",
    "run_experiment",
    "run_matrix",
]
__version__ = "0.1.0"
