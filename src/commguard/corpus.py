"""Explicit corpus membership and accepted-run allow-list contracts."""

from __future__ import annotations

import hashlib
import json
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
    workload_config_id: str | None = None

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
        _require(
            self.workload_config_id is None or bool(self.workload_config_id),
            "planned_run.workload_config_id: must be non-empty when provided",
        )

    def resolved_config_id(self) -> str:
        if self.workload_config_id is not None:
            return self.workload_config_id
        encoded = json.dumps(
            {
                "workload_family": self.workload_family,
                "config": self.config,
            },
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
        return f"config-{hashlib.sha256(encoded).hexdigest()[:16]}"


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
    calibration_reference: Mapping[str, Any] | None = None
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
        if self.calibration_reference is not None:
            from commguard.calibration import CALIBRATION_REFERENCE_FIELDS

            missing = sorted(set(CALIBRATION_REFERENCE_FIELDS) - set(self.calibration_reference))
            _require(
                not missing,
                f"corpus.calibration_reference: missing fields {missing}",
            )
            reference = self.calibration_reference
            path = str(reference["calibration_artifact_path"])
            _require(
                bool(path) and not path.startswith("/") and ".." not in path.split("/"),
                "corpus.calibration_reference.calibration_artifact_path: invalid",
            )
            _require(
                re.fullmatch(r"[0-9a-f]{64}", str(reference["calibration_sha256"])) is not None,
                "corpus.calibration_reference.calibration_sha256: invalid",
            )
            relationship = reference["calibration_relationship"]
            _require(
                relationship in {"current_session", "prior_session"},
                "corpus.calibration_reference.calibration_relationship: invalid",
            )
            _require(
                reference["calibration_created_in_current_session"]
                is (relationship == "current_session"),
                "corpus.calibration_reference: current/prior labels conflict",
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
            calibration_reference=data.get("calibration_reference"),
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
