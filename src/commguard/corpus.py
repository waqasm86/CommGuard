"""Explicit corpus membership and accepted-run allow-list contracts."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

from commguard.exceptions import ValidationError
from commguard.schemas import CURRENT_SCHEMA_VERSION


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


@dataclass(frozen=True)
class PlannedRun:
    """One declared corpus slot; an accepted evidence run may fill it once."""

    plan_id: str
    workload_family: str
    target_label: str
    config: Mapping[str, Any]
    designation: str = "benign"
    accepted_run_id: str | None = None
    random_seed: int = 1337

    def validate(self) -> None:
        _require(bool(self.plan_id), "planned_run.plan_id: required")
        _require(bool(self.workload_family), "planned_run.workload_family: required")
        _require(bool(self.target_label), "planned_run.target_label: required")
        _require(
            self.designation in {"benign", "adversarial", "calibration"},
            "planned_run.designation: invalid",
        )
        _require(isinstance(self.config, Mapping), "planned_run.config: must be an object")
        _require(self.random_seed >= 0, "planned_run.random_seed: must be non-negative")


@dataclass(frozen=True)
class CorpusManifest:
    corpus_id: str
    collection_id: str
    experiment_session_id: str
    node_id: str
    planned_runs: tuple[PlannedRun, ...]
    accepted_run_ids: tuple[str, ...]
    source_commit: str
    source_dirty: bool
    notebook_version: str | None
    input_archive_sha256: str | None
    random_seed: int
    schema_version: str = CURRENT_SCHEMA_VERSION
    artifact_kind: str = "corpus_manifest"

    def validate(self) -> None:
        _require(
            self.schema_version == CURRENT_SCHEMA_VERSION,
            "corpus.schema_version: unsupported version",
        )
        for name in (
            "corpus_id",
            "collection_id",
            "experiment_session_id",
            "node_id",
            "source_commit",
        ):
            _require(bool(getattr(self, name)), f"corpus.{name}: required")
        _require(bool(self.planned_runs), "corpus.planned_runs: must not be empty")
        _require(self.artifact_kind == "corpus_manifest", "corpus.artifact_kind: invalid")
        _require(isinstance(self.source_dirty, bool), "corpus.source_dirty: must be a boolean")
        _require(self.random_seed >= 0, "corpus.random_seed: must be non-negative")
        _require(
            self.input_archive_sha256 is None
            or re.fullmatch(r"[0-9a-f]{64}", self.input_archive_sha256) is not None,
            "corpus.input_archive_sha256: must be 64 lowercase hexadecimal characters",
        )
        plan_ids = [item.plan_id for item in self.planned_runs]
        _require(
            len(plan_ids) == len(set(plan_ids)),
            "corpus.planned_runs: plan IDs must be unique",
        )
        for item in self.planned_runs:
            item.validate()
        declared = [item.accepted_run_id for item in self.planned_runs if item.accepted_run_id]
        _require(
            len(declared) == len(set(declared)),
            "corpus.planned_runs: accepted run IDs must be unique",
        )
        _require(
            len(self.accepted_run_ids) == len(set(self.accepted_run_ids)),
            "corpus.accepted_run_ids: must be unique",
        )
        _require(
            set(self.accepted_run_ids) == set(declared),
            "corpus.accepted_run_ids: must exactly match planned-run assignments",
        )

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CorpusManifest:
        return cls(
            corpus_id=str(data.get("corpus_id", "")),
            collection_id=str(data.get("collection_id", "")),
            experiment_session_id=str(data.get("experiment_session_id", "")),
            node_id=str(data.get("node_id", "")),
            planned_runs=tuple(PlannedRun(**item) for item in data.get("planned_runs", ())),
            accepted_run_ids=tuple(str(value) for value in data.get("accepted_run_ids", ())),
            source_commit=str(data.get("source_commit", "")),
            source_dirty=data.get("source_dirty"),
            notebook_version=data.get("notebook_version"),
            input_archive_sha256=data.get("input_archive_sha256"),
            random_seed=int(data.get("random_seed", -1)),
            schema_version=str(data.get("schema_version", "")),
            artifact_kind=str(data.get("artifact_kind", "")),
        )

    def planned_for_run(self, run_id: str) -> PlannedRun | None:
        return next(
            (item for item in self.planned_runs if item.accepted_run_id == run_id),
            None,
        )

    def allows(self, run_id: str, designation: str) -> bool:
        """Require exact membership; calibration cannot leak into benign selection."""
        item = self.planned_for_run(run_id)
        return bool(item and run_id in self.accepted_run_ids and item.designation == designation)
