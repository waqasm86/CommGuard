from __future__ import annotations

import json
import tarfile

import pytest

from commguard.artifacts import ArtifactStore
from commguard.exceptions import ArtifactExistsError, ValidationError
from commguard.schemas import SCHEMA_VERSION


def artifact() -> dict[str, str]:
    return {
        "artifact_kind": "experiment_summary",
        "schema_version": SCHEMA_VERSION,
        "summary_type": "test",
    }


def test_create_only_json_and_hash(tmp_path) -> None:
    store = ArtifactStore(tmp_path)
    store.initialize()
    path = store.write_json("results/value.json", artifact())
    assert json.loads(path.read_text())["summary_type"] == "test"
    assert len(store.sha256("results/value.json")) == 64
    with pytest.raises(ArtifactExistsError):
        store.write_json("results/value.json", artifact())


def test_path_escape_is_rejected(tmp_path) -> None:
    store = ArtifactStore(tmp_path)
    with pytest.raises(ValidationError, match="escapes root"):
        store.resolve("../outside.json")


def test_export_contains_regular_artifacts(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    store.initialize()
    store.write_json("results/value.json", artifact())
    output = store.export(tmp_path / "bundle.tar.gz")
    with tarfile.open(output, "r:gz") as archive:
        assert "results/value.json" in archive.getnames()
