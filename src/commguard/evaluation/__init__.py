"""Grouped evaluation and transparent baseline models."""

from commguard.evaluation.baselines import evaluate_detector
from commguard.evaluation.splits import SplitPlan, audit_leakage, grouped_split, make_split_plan

__all__ = [
    "audit_leakage",
    "evaluate_detector",
    "grouped_split",
    "make_split_plan",
    "SplitPlan",
]
