#!/usr/bin/env python3
"""Fail on delivery-policy violations in tracked or pending repository files."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_FILE_BYTES = 1_000_000
PROHIBITED_PARTS = {
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "build",
    "dist",
}
PROHIBITED_SUFFIXES = (".pyc", ".whl", ".zip", ".tar.gz")


def candidate_paths() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return [ROOT / item.decode() for item in result.stdout.split(b"\0") if item]


def scan_text(path: Path, failures: list[str]) -> None:
    try:
        source = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return
    secret_patterns = {
        "AWS access key": r"AK" + r"IA[0-9A-Z]{16}",
        "GitHub token": r"gh" + r"[pousr]_[A-Za-z0-9_]{20,}",
        "OpenAI-style secret": r"sk" + r"-[A-Za-z0-9]{20,}",
        "private key": r"BEGIN " + r"(?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY",
    }
    for label, pattern in secret_patterns.items():
        if re.search(pattern, source):
            failures.append(f"{path.relative_to(ROOT)} contains a possible {label}")
    relative = path.relative_to(ROOT)
    if not str(relative).startswith(".agent/state/") and relative != Path(
        "tests/test_repository.py"
    ):
        personal_patterns = ("drive.google" + ".com", "/" + "home/", "/" + "media/")
        for pattern in personal_patterns:
            if pattern in source:
                failures.append(f"{relative} contains prohibited personal path/link {pattern}")


def check_notebooks(failures: list[str]) -> None:
    inventory_path = ROOT / "notebooks/canonical_notebooks.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    for name in inventory["canonical_notebooks"]:
        path = inventory_path.parent / name
        notebook = json.loads(path.read_text(encoding="utf-8"))
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            if cell.get("execution_count") is not None or cell.get("outputs"):
                failures.append(f"canonical notebook contains outputs: {name}")
        source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
        if "@" + "main" in source or 'checkout", "' + "main" in source:
            failures.append(f"canonical notebook uses a mutable main-branch install: {name}")


def main() -> int:
    failures: list[str] = []
    for path in candidate_paths():
        relative = path.relative_to(ROOT)
        if path.is_symlink():
            failures.append(f"symlink is not permitted in delivery: {relative}")
            continue
        if not path.is_file():
            continue
        if PROHIBITED_PARTS & set(relative.parts):
            failures.append(f"generated/cache path is not permitted: {relative}")
        if str(relative).endswith(PROHIBITED_SUFFIXES):
            failures.append(f"archive/build output is not permitted: {relative}")
        if path.stat().st_size > MAX_FILE_BYTES:
            failures.append(
                f"file exceeds {MAX_FILE_BYTES} bytes: {relative} ({path.stat().st_size})"
            )
        scan_text(path, failures)
    check_notebooks(failures)
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        return 1
    print("Delivery policy checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
