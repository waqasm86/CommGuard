"""Create-only artifact storage with hashing and safe export."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from commguard.exceptions import ArtifactExistsError, ValidationError
from commguard.schemas import validate_artifact

LAYOUT = ("environment", "corpora", "runs", "features", "splits", "results", "figures")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def restore_archive(
    archive: str | Path,
    destination: str | Path,
    *,
    expected_sha256: str,
    maximum_uncompressed_bytes: int = 10 * 1024**3,
    maximum_members: int = 100_000,
) -> Path:
    """Hash-check and safely restore a create-only CommGuard tar.gz archive."""
    archive_path = Path(archive).resolve()
    destination_path = Path(destination).resolve()
    if destination_path.exists():
        raise ArtifactExistsError(
            f"refusing to restore into an existing destination: {destination_path}"
        )
    if not archive_path.is_file():
        raise ValidationError(f"archive does not exist: {archive_path}")
    actual_sha256 = sha256_file(archive_path)
    if actual_sha256 != expected_sha256:
        raise ValidationError(
            f"archive SHA-256 mismatch: expected={expected_sha256} actual={actual_sha256}"
        )
    try:
        with tarfile.open(archive_path, mode="r:gz") as bundle:
            members = bundle.getmembers()
            if not members:
                raise ValidationError("archive contains no members")
            if len(members) > maximum_members:
                raise ValidationError(
                    f"archive member limit exceeded: {len(members)} > {maximum_members}"
                )
            total_size = sum(member.size for member in members if member.isfile())
            if total_size > maximum_uncompressed_bytes:
                raise ValidationError(
                    "archive uncompressed-size limit exceeded: "
                    f"{total_size} > {maximum_uncompressed_bytes}"
                )
            seen: set[PurePosixPath] = set()
            for member in members:
                relative = PurePosixPath(member.name)
                if (
                    not member.name
                    or relative.is_absolute()
                    or ".." in relative.parts
                    or relative in seen
                ):
                    raise ValidationError(f"unsafe or duplicate archive path: {member.name!r}")
                seen.add(relative)
                if member.issym() or member.islnk():
                    raise ValidationError(f"archive links are prohibited: {member.name!r}")
                if not (member.isfile() or member.isdir()):
                    raise ValidationError(f"unsupported archive member: {member.name!r}")
            destination_path.mkdir(parents=True, exist_ok=False)
            for member in members:
                target = destination_path.joinpath(*PurePosixPath(member.name).parts)
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                source = bundle.extractfile(member)
                if source is None:
                    raise ValidationError(f"archive file cannot be read: {member.name!r}")
                with source, target.open("xb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
    except BaseException:
        if destination_path.exists():
            shutil.rmtree(destination_path)
        raise
    return destination_path


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
        return sha256_file(self.resolve(relative))

    def export(self, output: str | Path) -> Path:
        """Export regular files without following symlinks."""
        output_path = Path(output).resolve()
        if output_path.exists():
            raise ArtifactExistsError(f"refusing to overwrite archive: {output_path}")
        if output_path == self.root or self.root in output_path.parents:
            raise ValidationError("export archive must be outside the artifact root")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
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
