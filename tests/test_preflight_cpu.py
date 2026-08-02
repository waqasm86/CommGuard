from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from commguard.environment import preflight
from commguard.exceptions import ReadinessError


def _stub_runtime_inventory(monkeypatch) -> None:
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
    _stub_runtime_inventory(monkeypatch)
    report = preflight.check_environment(strict=False)
    assert report["gpus"] == []
    assert report["strict_ready"] is False
    assert report["torch"]["available"] is False


@pytest.mark.parametrize("nested", [False, True])
def test_preflight_creates_missing_output_directory(tmp_path, monkeypatch, nested) -> None:
    _stub_runtime_inventory(monkeypatch)
    output = tmp_path / "missing"
    if nested:
        output = output / "nested" / "artifacts"

    report = preflight.check_environment(output=output)

    assert output.is_dir()
    assert report["resources"]["disk_total_bytes"] > 0
    assert len(list((output / "environment").glob("preflight-*.json"))) == 1


def test_preflight_preserves_existing_output_directory(tmp_path, monkeypatch) -> None:
    _stub_runtime_inventory(monkeypatch)
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("preserve\n", encoding="utf-8")

    preflight.check_environment(output=output)

    assert sentinel.read_text(encoding="utf-8") == "preserve\n"


def test_preflight_rejects_output_file(tmp_path) -> None:
    output = tmp_path / "not-a-directory"
    output.write_text("file\n", encoding="utf-8")

    with pytest.raises(ReadinessError, match="not a directory"):
        preflight.check_environment(output=output)


def test_preflight_reports_nested_directory_creation_failure(tmp_path) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("file\n", encoding="utf-8")

    with pytest.raises(ReadinessError, match="cannot create preflight output directory"):
        preflight.check_environment(output=blocker / "child")


def test_preflight_output_none_uses_current_directory(monkeypatch) -> None:
    _stub_runtime_inventory(monkeypatch)
    observed: list[Path] = []
    real_disk_usage = shutil.disk_usage

    def disk_usage(path):
        observed.append(Path(path))
        return real_disk_usage(path)

    monkeypatch.setattr(preflight.shutil, "disk_usage", disk_usage)

    preflight.check_environment(output=None)

    assert observed == [Path.cwd().resolve()]
