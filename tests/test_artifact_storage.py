from __future__ import annotations

import pytest

from commguard.artifacts.storage import ArtifactStore
from commguard.exceptions import ArtifactExistsError


def test_storage_rejects_overwrite_by_default(tmp_path) -> None:
    store = ArtifactStore(tmp_path)
    first = store.write_text("runs/run-1/raw.log", "first\n")
    assert first.read_text(encoding="utf-8") == "first\n"
    with pytest.raises(ArtifactExistsError, match="refusing to overwrite"):
        store.write_text("runs/run-1/raw.log", "replacement\n")
    assert first.read_text(encoding="utf-8") == "first\n"
