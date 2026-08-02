"""Create-only artifact storage with hashing and safe export."""

from __future__ import annotations

import hashlib
import json
import os
import tarfile
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from commguard.exceptions import ArtifactExistsError, ValidationError
from commguard.schemas import validate_artifact

LAYOUT = ("environment", "corpora", "runs", "features", "splits", "results", "figures")


class ArtifactStore:
    """Human-inspectable, create-only evidence tree."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()

    def initialize(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        for directory in LAYOUT:
            (self.root / directory).mkdir(exist_ok=True)

    def resolve(self, relative: str | Path) -> Path:
        path = (self.root / relative).resolve()
        if path != self.root and self.root not in path.parents:
            raise ValidationError(f"artifact path escapes root: {relative}")
        return path

    def write_json(
        self,
        relative: str | Path,
        data: Mapping[str, Any],
        validate: bool = True,
    ) -> Path:
        if validate:
            validate_artifact(data)
        encoded = (json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
        return self._exclusive_write(relative, encoded)

    def write_jsonl(
        self,
        relative: str | Path,
        records: Iterable[Mapping[str, Any]],
        validate: bool = True,
    ) -> Path:
        lines: list[str] = []
        for record in records:
            if validate:
                validate_artifact(record)
            lines.append(json.dumps(record, sort_keys=True, allow_nan=False))
        return self._exclusive_write(relative, ("\n".join(lines) + "\n").encode())

    def write_text(self, relative: str | Path, text: str) -> Path:
        return self._exclusive_write(relative, text.encode("utf-8"))

    def _exclusive_write(self, relative: str | Path, content: bytes) -> Path:
        target = self.resolve(relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError as exc:
            raise ArtifactExistsError(f"refusing to overwrite artifact: {target}") from exc
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
        except BaseException:
            target.unlink(missing_ok=True)
            raise
        return target

    def sha256(self, relative: str | Path) -> str:
        digest = hashlib.sha256()
        with self.resolve(relative).open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def export(self, output: str | Path) -> Path:
        """Export regular files without following symlinks."""
        output_path = Path(output).resolve()
        if output_path.exists():
            raise ArtifactExistsError(f"refusing to overwrite archive: {output_path}")
        if output_path == self.root or self.root in output_path.parents:
            raise ValidationError("export archive must be outside the artifact root")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=output_path.parent, delete=False) as temporary:
            temp_path = Path(temporary.name)
        try:
            with tarfile.open(temp_path, mode="w:gz") as archive:
                for path in sorted(self.root.rglob("*")):
                    if path.is_file() and not path.is_symlink():
                        archive.add(path, arcname=path.relative_to(self.root), recursive=False)
            os.replace(temp_path, output_path)
        finally:
            temp_path.unlink(missing_ok=True)
        return output_path
