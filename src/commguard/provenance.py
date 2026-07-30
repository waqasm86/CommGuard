"""Source-state identifiers that work inside uploaded Kaggle datasets."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path


def source_identifier() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
        timeout=2,
    )
    if result.returncode == 0 and result.stdout.strip():
        return result.stdout.strip()
    root = Path(__file__).resolve().parents[2]
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
