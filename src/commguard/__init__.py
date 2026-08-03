"""CommGuard's stable public API."""

from commguard.adversarial import ADVERSARIAL_STRATEGIES, AdversarialHoldoutPlan
from commguard.corpus import CorpusManifest, PlannedRun
from commguard.environment import preflight
from commguard.environment.preflight import check_environment
from commguard.evaluation import evaluate_detector
from commguard.features import extract_feature_result, extract_features, require_primary_coverage
from commguard.orchestrator import (
    run_adversarial_matrix,
    run_experiment,
    run_matrix,
    run_periodic_synchronization_study,
    run_segmented_series,
)
from commguard.provenance import ProvenanceContext
from commguard.reporting import generate_report
from commguard.schemas import load_artifact
from commguard.scope import PROTOTYPE_SCOPE_DECLARATION, PROTOTYPE_SCOPE_METADATA
from commguard.workloads import list_workloads

__all__ = [
    "ADVERSARIAL_STRATEGIES",
    "AdversarialHoldoutPlan",
    "check_environment",
    "CorpusManifest",
    "evaluate_detector",
    "extract_feature_result",
    "extract_features",
    "generate_report",
    "list_workloads",
    "load_artifact",
    "PlannedRun",
    "PROTOTYPE_SCOPE_DECLARATION",
    "PROTOTYPE_SCOPE_METADATA",
    "preflight",
    "ProvenanceContext",
    "require_primary_coverage",
    "run_adversarial_matrix",
    "run_experiment",
    "run_matrix",
    "run_periodic_synchronization_study",
    "run_segmented_series",
]
__version__ = "0.2.0"
