"""Stable source and experiment identifiers, including uploaded datasets."""

from __future__ import annotations

import hashlib
import re
import socket
import subprocess
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_ID_COMPONENT = re.compile(r"[^a-zA-Z0-9_.-]+")


@dataclass(frozen=True)
class SourceState:
    commit: str
    dirty: bool


def _identifier_component(value: str) -> str:
    component = _ID_COMPONENT.sub("-", value.strip()).strip("-.")
    if not component:
        raise ValueError("identifier component must contain a letter or number")
    return component


def _run_git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        text=True,
        check=False,
        timeout=2,
    )


def _tree_identifier(root: Path) -> str:
    digest = hashlib.sha256()
    candidates = [root / "pyproject.toml", root / "README.md"]
    candidates.extend(sorted((root / "src").rglob("*.py")))
    for path in candidates:
        if not path.is_file():
            continue
        digest.update(str(path.relative_to(root)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return f"tree-sha256:{digest.hexdigest()}"


def source_state(root: str | Path | None = None) -> SourceState:
    repository = Path(root).resolve() if root else Path(__file__).resolve().parents[2]
    revision = _run_git(repository, "rev-parse", "HEAD")
    status = _run_git(repository, "status", "--porcelain")
    if revision.returncode == 0 and revision.stdout.strip():
        dirty = status.returncode != 0 or bool(status.stdout.strip())
        return SourceState(revision.stdout.strip(), dirty)
    return SourceState(_tree_identifier(repository), True)


def source_identifier() -> str:
    """Backward-compatible source identifier."""
    return source_state().commit


@dataclass(frozen=True)
class ProvenanceContext:
    experiment_session_id: str
    collection_id: str
    corpus_id: str
    node_id: str
    source_commit: str
    source_dirty: bool
    notebook_version: str | None
    input_archive_sha256: str | None
    random_seed: int

    def __post_init__(self) -> None:
        for name in (
            "experiment_session_id",
            "collection_id",
            "corpus_id",
            "node_id",
            "source_commit",
        ):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} must be non-empty")
        if self.random_seed < 0:
            raise ValueError("random_seed must be non-negative")
        if self.input_archive_sha256 is not None and not re.fullmatch(
            r"[0-9a-f]{64}", self.input_archive_sha256
        ):
            raise ValueError("input_archive_sha256 must be 64 lowercase hexadecimal characters")

    @classmethod
    def create(
        cls,
        *,
        corpus_id: str,
        collection_id: str | None = None,
        experiment_session_id: str | None = None,
        node_id: str | None = None,
        notebook_version: str | None = None,
        input_archive_sha256: str | None = None,
        random_seed: int = 1337,
        repository_root: str | Path | None = None,
    ) -> ProvenanceContext:
        state = source_state(repository_root)
        return cls(
            experiment_session_id=(experiment_session_id or f"session-{uuid.uuid4().hex}"),
            collection_id=collection_id or f"collection-{uuid.uuid4().hex}",
            corpus_id=corpus_id,
            node_id=node_id or socket.gethostname(),
            source_commit=state.commit,
            source_dirty=state.dirty,
            notebook_version=notebook_version,
            input_archive_sha256=input_archive_sha256,
            random_seed=random_seed,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def run_fields(self, *, random_seed: int | None = None) -> dict[str, Any]:
        values = self.to_dict()
        values["random_seed"] = self.random_seed if random_seed is None else random_seed
        return values


def new_run_id(name: str, experiment_session_id: str) -> str:
    """Generate a unique run ID while retaining its owning session for inspection."""
    session_suffix = experiment_session_id.removeprefix("session-")[:8]
    return (
        f"run-{_identifier_component(session_suffix)}-"
        f"{_identifier_component(name)}-{uuid.uuid4().hex[:12]}"
    )


def new_corpus_id(name: str) -> str:
    """Generate a corpus identifier for one deliberate experiment matrix."""
    return f"corpus-{_identifier_component(name)}-{uuid.uuid4().hex[:16]}"
