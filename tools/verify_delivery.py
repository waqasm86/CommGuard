#!/usr/bin/env python3
"""Fail on delivery-policy violations in tracked or pending repository files."""

from __future__ import annotations

import json
import re
import subprocess
import sys
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
PERSONAL_PATH_PATTERN = re.compile(
    r"(?:file://)?/(?:home|media)/[^\s<>)\]}\"']+",
    flags=re.IGNORECASE,
)


def candidate_paths(root: Path = ROOT) -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    return [root / item.decode() for item in result.stdout.split(b"\0") if item]


def scan_text(path: Path, failures: list[str], *, root: Path = ROOT) -> None:
    try:
        source = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return
    secret_patterns = {
        "AWS access key": r"AK" + r"IA[0-9A-Z]{16}",
        "GitHub token": r"gh" + r"[pousr]_[A-Za-z0-9_]{20,}",
        "GitHub fine-grained token": r"github" + r"_pat_[A-Za-z0-9_]{20,}",
        "OpenAI-style secret": r"sk" + r"-[A-Za-z0-9]{20,}",
        "private key": r"BEGIN " + r"(?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY",
    }
    for label, pattern in secret_patterns.items():
        if re.search(pattern, source):
            failures.append(f"{path.relative_to(root)} contains a possible {label}")
    relative = path.relative_to(root)
    personal_path = PERSONAL_PATH_PATTERN.search(source)
    if personal_path:
        failures.append(
            f"{relative} contains prohibited personal path/link {personal_path.group(0)}"
        )


def check_notebooks(failures: list[str], *, root: Path = ROOT) -> None:
    inventory_path = root / "notebooks/canonical_notebooks.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    entries = inventory["canonical_notebooks"]
    names = [entry["filename"] if isinstance(entry, dict) else entry for entry in entries]
    if names != [
        "commguard_calibration_v4.ipynb",
        "commguard_benign_corpus_v2.ipynb",
        "commguard_detector_evaluation_v2.ipynb",
        "commguard_adversarial_redteam_v1.ipynb",
    ]:
        failures.append("canonical notebook inventory order is invalid")
    historical_entries = inventory.get("historical_notebooks", [])
    historical_names = [
        entry["filename"] if isinstance(entry, dict) else entry for entry in historical_entries
    ]
    if historical_names != ["commguard_calibration_v3.ipynb"]:
        failures.append("historical notebook inventory is invalid")

    for name in [*names, *historical_names]:
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


def scan_repository(root: Path = ROOT, *, include_notebooks: bool = True) -> list[str]:
    failures: list[str] = []
    for path in candidate_paths(root):
        relative = path.relative_to(root)
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
        scan_text(path, failures, root=root)
    if include_notebooks:
        check_notebooks(failures, root=root)
    return failures


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    root = ROOT
    include_notebooks = True
    if args:
        if len(args) != 2 or args[0] != "--root":
            raise SystemExit("usage: verify_delivery.py [--root PATH]")
        root = Path(args[1]).resolve()
        include_notebooks = (root / "notebooks/canonical_notebooks.json").is_file()
    failures = scan_repository(root, include_notebooks=include_notebooks)
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        return 1
    print("Delivery policy checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
