from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from commguard.environment import preflight


def test_import_does_not_eagerly_load_gpu_libraries() -> None:
    root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root / "src")
    code = (
        "import sys; import commguard.environment.preflight; "
        "assert 'torch' not in sys.modules; assert 'pynvml' not in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )
    assert result.returncode == 0, result.stderr


def test_cpu_preflight_reports_capability_without_fallback(monkeypatch) -> None:
    unavailable = {"available": False, "error": "not installed"}
    monkeypatch.setattr(preflight, "_gpu_inventory", lambda: ([], {"exit_code": None}))
    monkeypatch.setattr(preflight, "_torch_details", lambda: unavailable)
    monkeypatch.setattr(preflight, "_telemetry_capabilities", lambda: unavailable)
    monkeypatch.setattr(
        preflight,
        "_command",
        lambda args, timeout=10: {
            "command": args,
            "exit_code": None,
            "stdout": "",
            "stderr": "not installed",
        },
    )
    report = preflight.check_environment(strict=False)
    assert report["gpus"] == []
    assert report["strict_ready"] is False
    assert report["torch"]["available"] is False
