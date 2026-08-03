"""Grouped evaluation and transparent baseline models."""

from commguard.evaluation.baselines import evaluate_detector
from commguard.evaluation.splits import (
    CrossValidationFold,
    SplitPlan,
    audit_leakage,
    group_cross_validation_folds,
    grouped_split,
    make_split_plan,
)

__all__ = [
    "audit_leakage",
    "evaluate_detector",
    "group_cross_validation_folds",
    "grouped_split",
    "make_split_plan",
    "SplitPlan",
    "CrossValidationFold",
]
