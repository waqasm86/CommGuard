"""Validated whole-run split hierarchy for primary and robustness evaluation."""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

from commguard.features import METADATA_COLUMNS

SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class RunGroup:
    run_id: str
    experiment_session_id: str
    target_label: str
    workload_family: str
    workload_config_id: str


@dataclass(frozen=True)
class SplitPlan:
    assignments: dict[str, str]
    actual_strategy: str
    requested_mode: str
    seed: int
    diagnostic_only: bool
    run_ids_by_split: dict[str, list[str]]
    session_ids_by_split: dict[str, list[str]]
    class_run_counts_by_split: dict[str, dict[str, int]]
    family_run_counts_by_split: dict[str, dict[str, int]]
    config_run_counts_by_split: dict[str, dict[str, int]]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _run_groups(rows: Iterable[Mapping[str, Any]]) -> dict[str, RunGroup]:
    groups: dict[str, RunGroup] = {}
    for row in rows:
        run_id = str(row.get("run_id", ""))
        if not run_id:
            raise ValueError("feature row is missing run_id")
        if row.get("legacy_grouping_ambiguous") is True:
            raise ValueError(
                f"run_id {run_id!r} has ambiguous legacy grouping; "
                "a true experiment_session_id is required"
            )
        session = row.get("experiment_session_id")
        if not session:
            raise ValueError(
                f"run_id {run_id!r} has no true experiment_session_id; "
                "environment/session_fingerprint values cannot be used for grouping"
            )
        candidate = RunGroup(
            run_id=run_id,
            experiment_session_id=str(session),
            target_label=str(row.get("target_label", "")),
            workload_family=str(row.get("workload_family", "")),
            workload_config_id=str(row.get("workload_config_id") or row.get("plan_id") or ""),
        )
        if not candidate.target_label or not candidate.workload_family:
            raise ValueError(f"run_id {run_id!r} is missing target/family metadata")
        if not candidate.workload_config_id:
            raise ValueError(f"run_id {run_id!r} is missing workload configuration identity")
        previous = groups.get(run_id)
        if previous is not None and previous != candidate:
            changed = [
                name
                for name in (
                    "experiment_session_id",
                    "target_label",
                    "workload_family",
                    "workload_config_id",
                )
                if getattr(previous, name) != getattr(candidate, name)
            ]
            raise ValueError(f"run_id {run_id!r} spans multiple {', '.join(changed)} values")
        groups[run_id] = candidate
    if not groups:
        raise ValueError("cannot split an empty feature collection")
    return groups


def _counts(
    groups: Mapping[str, RunGroup], assignments: Mapping[str, str], field: str
) -> dict[str, dict[str, int]]:
    result: dict[str, Counter[str]] = {split: Counter() for split in SPLITS}
    for run_id, group in groups.items():
        result[assignments[run_id]][str(getattr(group, field))] += 1
    return {split: dict(sorted(result[split].items())) for split in SPLITS}


def _finalize(
    groups: Mapping[str, RunGroup],
    assignments: dict[str, str],
    *,
    actual_strategy: str,
    requested_mode: str,
    seed: int,
    diagnostic_only: bool,
    required_families: tuple[str, ...],
) -> SplitPlan:
    if set(assignments) != set(groups):
        raise ValueError("split assignments do not cover every run exactly once")
    unknown = sorted(set(assignments.values()) - set(SPLITS))
    if unknown:
        raise ValueError(f"split assignments contain invalid values: {unknown}")
    run_ids = {
        split: sorted(run_id for run_id, value in assignments.items() if value == split)
        for split in SPLITS
    }
    class_counts = _counts(groups, assignments, "target_label")
    warnings: list[str] = []
    if diagnostic_only:
        if not run_ids["train"] or not run_ids["test"]:
            raise ValueError("diagnostic holdout requires non-empty train and test groups")
    else:
        missing_splits = [split for split in SPLITS if not run_ids[split]]
        if missing_splits:
            raise ValueError(f"primary split has empty partitions: {missing_splits}")
        expected_classes = set(group.target_label for group in groups.values())
        if len(expected_classes) < 2:
            raise ValueError("primary evaluation requires at least two target classes")
        invalid_classes = {
            split: sorted(expected_classes - set(class_counts[split]))
            for split in SPLITS
            if set(class_counts[split]) != expected_classes
        }
        if invalid_classes:
            raise ValueError(f"primary split lacks target-class coverage: {invalid_classes}")

    family_counts = _counts(groups, assignments, "workload_family")
    missing_families_by_split: dict[str, list[str]] = {}
    for split in SPLITS:
        missing = sorted(set(required_families) - set(family_counts[split]))
        if missing:
            missing_families_by_split[split] = missing
    if missing_families_by_split and not diagnostic_only:
        raise ValueError(
            f"primary split lacks required-family coverage: {missing_families_by_split}"
        )
    for split, missing in missing_families_by_split.items():
        warnings.append(f"{split} lacks required families: {missing}")
    if diagnostic_only:
        one_class = [split for split in SPLITS if run_ids[split] and len(class_counts[split]) < 2]
        if one_class:
            warnings.append(
                f"diagnostic holdout has one-class partitions {one_class}; "
                "balanced metrics are undefined"
            )

    sessions: dict[str, set[str]] = {split: set() for split in SPLITS}
    for run_id, split in assignments.items():
        sessions[split].add(groups[run_id].experiment_session_id)
    return SplitPlan(
        assignments=dict(sorted(assignments.items())),
        actual_strategy=actual_strategy,
        requested_mode=requested_mode,
        seed=seed,
        diagnostic_only=diagnostic_only,
        run_ids_by_split=run_ids,
        session_ids_by_split={split: sorted(values) for split, values in sessions.items()},
        class_run_counts_by_split=class_counts,
        family_run_counts_by_split=family_counts,
        config_run_counts_by_split=_counts(groups, assignments, "workload_config_id"),
        warnings=warnings,
    )


def _stratified_assignments(
    groups: Mapping[str, RunGroup],
    seed: int,
    test_fraction: float,
    validation_fraction: float,
    required_families: tuple[str, ...],
) -> dict[str, str]:
    strata: dict[str, list[str]] = defaultdict(list)
    for run_id, group in groups.items():
        key = (
            f"family:{group.workload_family}"
            if group.workload_family in required_families
            else f"class:{group.target_label}"
        )
        strata[key].append(run_id)
    assignments: dict[str, str] = {}
    for stratum, run_ids in sorted(strata.items()):
        random.Random(f"{seed}:{stratum}").shuffle(run_ids)
        if len(run_ids) < 3:
            raise ValueError(
                f"split stratum {stratum!r} has {len(run_ids)} runs; "
                "at least three are required for train/validation/test"
            )
        test_count = max(1, round(len(run_ids) * test_fraction))
        validation_count = max(1, round(len(run_ids) * validation_fraction))
        while test_count + validation_count >= len(run_ids):
            if validation_count > 1:
                validation_count -= 1
            elif test_count > 1:
                test_count -= 1
            else:
                raise ValueError(f"cannot create three partitions for split stratum {stratum!r}")
        for index, run_id in enumerate(run_ids):
            assignments[run_id] = (
                "test"
                if index < test_count
                else "validation"
                if index < test_count + validation_count
                else "train"
            )
    return assignments


def _session_assignments(
    groups: Mapping[str, RunGroup],
    seed: int,
    required_families: tuple[str, ...],
) -> dict[str, str] | None:
    sessions = sorted({group.experiment_session_id for group in groups.values()})
    if len(sessions) < 3:
        return None
    candidates = [
        (test, validation) for test in sessions for validation in sessions if test != validation
    ]
    random.Random(seed).shuffle(candidates)
    expected_classes = {group.target_label for group in groups.values()}
    for test_session, validation_session in candidates:
        assignments = {
            run_id: (
                "test"
                if group.experiment_session_id == test_session
                else "validation"
                if group.experiment_session_id == validation_session
                else "train"
            )
            for run_id, group in groups.items()
        }
        valid = True
        for split in SPLITS:
            selected = [groups[run_id] for run_id, value in assignments.items() if value == split]
            classes = {group.target_label for group in selected}
            families = {group.workload_family for group in selected}
            if classes != expected_classes or not set(required_families).issubset(families):
                valid = False
                break
        if valid:
            return assignments
    return None


def _diagnostic_holdout_assignments(
    groups: Mapping[str, RunGroup],
    *,
    field: str,
    value: str,
    seed: int,
) -> dict[str, str]:
    heldout = {run_id for run_id, group in groups.items() if str(getattr(group, field)) == value}
    if not heldout:
        raise ValueError(f"holdout value {value!r} is not present in {field}")
    remaining = [run_id for run_id in groups if run_id not in heldout]
    if not remaining:
        raise ValueError("holdout consumed every run")
    by_label: dict[str, list[str]] = defaultdict(list)
    for run_id in remaining:
        by_label[groups[run_id].target_label].append(run_id)
    assignments = {run_id: "test" for run_id in heldout}
    for label, run_ids in sorted(by_label.items()):
        random.Random(f"{seed}:{field}:{value}:{label}").shuffle(run_ids)
        validation_count = 1 if len(run_ids) >= 2 else 0
        for index, run_id in enumerate(run_ids):
            assignments[run_id] = "validation" if index < validation_count else "train"
    return assignments


def make_split_plan(
    rows: Iterable[Mapping[str, Any]],
    *,
    mode: str = "auto",
    seed: int = 20260730,
    test_fraction: float = 0.2,
    validation_fraction: float = 0.2,
    required_families: tuple[str, ...] = (),
    holdout_value: str | None = None,
) -> SplitPlan:
    """Create and validate the requested split without consulting feature values."""
    if not 0 < test_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("primary split fractions must satisfy 0 < test, validation < 1")
    if test_fraction + validation_fraction >= 1:
        raise ValueError("test and validation fractions must sum to less than 1")
    groups = _run_groups(rows)
    supported_modes = {
        "auto",
        "session_holdout",
        "stratified_whole_run",
        "family_holdout",
        "configuration_holdout",
    }
    if mode not in supported_modes:
        raise ValueError(f"unknown split mode {mode!r}; expected {sorted(supported_modes)}")

    if mode in {"auto", "session_holdout"}:
        assignments = _session_assignments(groups, seed, required_families)
        if assignments is not None:
            return _finalize(
                groups,
                assignments,
                actual_strategy="independent_session_holdout",
                requested_mode=mode,
                seed=seed,
                diagnostic_only=False,
                required_families=required_families,
            )
        if mode == "session_holdout":
            raise ValueError(
                "no valid independent-session holdout preserves classes and required families "
                "in train/validation/test"
            )

    if mode in {"auto", "stratified_whole_run"}:
        assignments = _stratified_assignments(
            groups,
            seed,
            test_fraction,
            validation_fraction,
            required_families,
        )
        return _finalize(
            groups,
            assignments,
            actual_strategy="deterministic_class_stratified_whole_run",
            requested_mode=mode,
            seed=seed,
            diagnostic_only=False,
            required_families=required_families,
        )

    if holdout_value is None:
        raise ValueError(f"split mode {mode!r} requires holdout_value")
    field = "workload_family" if mode == "family_holdout" else "workload_config_id"
    assignments = _diagnostic_holdout_assignments(
        groups,
        field=field,
        value=holdout_value,
        seed=seed,
    )
    return _finalize(
        groups,
        assignments,
        actual_strategy=mode,
        requested_mode=mode,
        seed=seed,
        diagnostic_only=True,
        required_families=required_families,
    )


def grouped_split(
    rows: Iterable[Mapping[str, Any]],
    seed: int = 20260730,
    test_fraction: float = 0.2,
    validation_fraction: float = 0.2,
) -> dict[str, str]:
    """Compatibility wrapper returning validated whole-run assignments."""
    return make_split_plan(
        rows,
        seed=seed,
        test_fraction=test_fraction,
        validation_fraction=validation_fraction,
    ).assignments


def audit_leakage(
    rows: Iterable[Mapping[str, Any]],
    feature_columns: Iterable[str],
    assignments: Mapping[str, str],
) -> dict[str, Any]:
    records = list(rows)
    columns = list(feature_columns)
    forbidden_tokens = ("run_id", "timestamp", "path", "filename", "label", "family", "session")
    forbidden = [
        column for column in columns if any(token in column.lower() for token in forbidden_tokens)
    ]
    grouped: dict[str, set[str]] = defaultdict(set)
    for record in records:
        run_id = str(record["run_id"])
        grouped[run_id].add(str(assignments[run_id]))
    split_leaks = {run_id: sorted(splits) for run_id, splits in grouped.items() if len(splits) != 1}
    durations_by_label: dict[str, set[float]] = defaultdict(set)
    for record in records:
        durations_by_label[str(record["target_label"])].add(float(record["window_seconds"]))
    duration_exclusive = len({tuple(sorted(values)) for values in durations_by_label.values()}) > 1
    ambiguous_runs = sorted(
        {
            str(record["run_id"])
            for record in records
            if record.get("legacy_grouping_ambiguous") is True
            or not record.get("experiment_session_id")
        }
    )
    passed = not forbidden and not split_leaks and not duration_exclusive and not ambiguous_runs
    return {
        "passed": passed,
        "forbidden_feature_columns": forbidden,
        "run_split_leaks": split_leaks,
        "class_specific_window_lengths": duration_exclusive,
        "ambiguous_or_missing_session_runs": ambiguous_runs,
        "excluded_metadata_columns": sorted(METADATA_COLUMNS),
    }
