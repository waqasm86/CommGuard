from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import commguard

ROOT = Path(__file__).resolve().parents[1]


def test_public_package_version_is_importable() -> None:
    assert commguard.__version__ == "0.1.0"


def test_module_help_lists_phase_one_commands() -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "src"), environment.get("PYTHONPATH", "")]
    )
    result = subprocess.run(
        [sys.executable, "-m", "commguard", "--help"],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )
    assert result.returncode == 0
    for command in ("preflight", "calibrate", "run", "features", "evaluate", "report"):
        assert command in result.stdout
