"""Kaggle dual-T4 environment inventory and strict readiness gates."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from commguard.artifacts import ArtifactStore
from commguard.exceptions import ReadinessError
from commguard.provenance import source_identifier
from commguard.schemas import SCHEMA_VERSION

PACKAGES = (
    "torch",
    "nvidia-ml-py",
    "nvidia-nccl-cu12",
    "numpy",
    "pandas",
    "scikit-learn",
    "pyarrow",
    "transformers",
    "accelerate",
    "datasets",
    "peft",
    "psutil",
)


def _command(args: list[str], timeout: float = 10.0) -> dict[str, Any]:
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=False)
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return {"command": args, "exit_code": None, "stdout": "", "stderr": str(exc)}
    return {
        "command": args,
        "exit_code": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def _gpu_inventory() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    query = _command(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,pci.bus_id,memory.total,driver_version,compute_cap",
            "--format=csv,noheader,nounits",
        ]
    )
    gpus: list[dict[str, Any]] = []
    if query["exit_code"] == 0:
        for line in query["stdout"].splitlines():
            fields = [item.strip() for item in line.split(",")]
            if len(fields) != 7:
                continue
            gpus.append(
                {
                    "index": int(fields[0]),
                    "name": fields[1],
                    "uuid": fields[2],
                    "pci_bus_id": fields[3],
                    "memory_total_mib": int(fields[4]),
                    "driver_version": fields[5],
                    "compute_capability": fields[6],
                }
            )
    return gpus, query


def _torch_details() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    nccl_version: str | None = None
    try:
        raw = torch.cuda.nccl.version()
        nccl_version = ".".join(map(str, raw)) if isinstance(raw, tuple) else str(raw)
    except Exception:
        pass
    peer_access: dict[str, bool | None] = {}
    for source, target in ((0, 1), (1, 0)):
        try:
            peer_access[f"{source}->{target}"] = bool(
                torch.cuda.can_device_access_peer(source, target)
            )
        except Exception:
            peer_access[f"{source}->{target}"] = None
    return {
        "available": True,
        "version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_runtime": torch.version.cuda,
        "device_count": int(torch.cuda.device_count()),
        "distributed_available": bool(torch.distributed.is_available()),
        "nccl_available": bool(torch.distributed.is_nccl_available()),
        "nccl_version": nccl_version,
        "cudnn_version": torch.backends.cudnn.version(),
        "peer_access": peer_access,
    }


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _telemetry_capabilities() -> dict[str, Any]:
    try:
        from commguard.telemetry import NvmlBackend

        backend = NvmlBackend()
        backend.initialize()
        try:
            devices = []
            for index in range(backend.device_count()):
                readings = backend.read_fields(index)
                devices.append(
                    {
                        "gpu_index": index,
                        "gpu_uuid": backend.device_uuid(index),
                        "fields": {
                            name: {
                                "value": reading.value,
                                "supported": reading.supported,
                                "unit": reading.unit,
                                "error": reading.error,
                            }
                            for name, reading in readings.items()
                        },
                    }
                )
            return {"available": True, "devices": devices, "error": None}
        finally:
            backend.shutdown()
    except Exception as exc:
        return {
            "available": False,
            "devices": [],
            "error": f"{type(exc).__name__}: {exc}",
        }


def _network_status(check: bool) -> dict[str, Any]:
    if not check:
        return {"checked": False, "reachable": None, "reason": "not requested"}
    try:
        with socket.create_connection(("pypi.org", 443), timeout=2):
            return {"checked": True, "reachable": True, "reason": None}
    except OSError as exc:
        return {"checked": True, "reachable": False, "reason": str(exc)}


def check_environment(
    strict: bool = False,
    output: str | Path | None = None,
    check_network: bool = False,
) -> dict[str, Any]:
    """Inspect runtime facts; strict mode requires exactly two NVIDIA T4s and NCCL."""
    gpus, gpu_query = _gpu_inventory()
    torch_info = _torch_details()
    topology = _command(["nvidia-smi", "topo", "-m"])
    nvcc = _command(["nvcc", "--version"])
    telemetry_capabilities = _telemetry_capabilities()
    disk = shutil.disk_usage(Path(output or ".").resolve())
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        pages = os.sysconf("SC_PHYS_PAGES")
        ram_total = int(page_size * pages)
    except (ValueError, OSError, AttributeError):
        ram_total = None

    readiness = {
        "exactly_two_gpus": len(gpus) == 2,
        "both_t4": len(gpus) == 2 and all("T4" in gpu["name"] for gpu in gpus),
        "torch_cuda": bool(torch_info.get("cuda_available")),
        "torch_sees_two": torch_info.get("device_count") == 2,
        "nccl_available": bool(torch_info.get("nccl_available")),
        "distinct_gpu_uuids": len({gpu["uuid"] for gpu in gpus}) == 2,
        "topology_command_supported": topology["exit_code"] == 0,
    }
    strict_ready = all(
        readiness[key]
        for key in (
            "exactly_two_gpus",
            "both_t4",
            "torch_cuda",
            "torch_sees_two",
            "nccl_available",
            "distinct_gpu_uuids",
        )
    )
    fingerprint_input = {
        "gpus": gpus,
        "platform": platform.platform(),
        "kernel": platform.release(),
        "torch": torch_info,
        "packages": _package_versions(),
        "hostname": socket.gethostname(),
        "kaggle_session": {
            key: os.environ.get(key)
            for key in ("KAGGLE_KERNEL_RUN_TYPE", "KAGGLE_KERNEL_INTEGRATIONS", "KAGGLE_URL_BASE")
        },
        "telemetry_capabilities": telemetry_capabilities,
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_input, sort_keys=True, default=str).encode()
    ).hexdigest()
    report: dict[str, Any] = {
        "artifact_kind": "environment_report",
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "session_fingerprint": fingerprint,
        "source_commit": source_identifier(),
        "python": {"version": sys.version, "executable": sys.executable},
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "hostname": socket.gethostname(),
        },
        "resources": {
            "ram_total_bytes": ram_total,
            "disk_total_bytes": disk.total,
            "disk_free_bytes": disk.free,
        },
        "gpus": gpus,
        "nvidia_smi_query": gpu_query,
        "topology": topology,
        "nvcc": nvcc,
        "torch": torch_info,
        "telemetry_capabilities": telemetry_capabilities,
        "packages": fingerprint_input["packages"],
        "network": _network_status(check_network),
        "readiness": readiness,
        "strict_ready": strict_ready,
        "limitations": [
            "GPU visibility alone does not prove two-rank participation.",
            "Peer access and topology support do not prove PCIe counters are informative.",
            "A torchrun NCCL smoke test and communication calibration are still required.",
        ],
    }
    if output is not None:
        store = ArtifactStore(output)
        store.initialize()
        store.write_json(
            "environment/preflight-"
            f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}.json",
            report,
        )
    if strict and not strict_ready:
        failed = [name for name, passed in readiness.items() if not passed]
        raise ReadinessError("dual-T4 preflight failed: " + ", ".join(failed))
    return report


def summarize_environment(report: dict[str, Any]) -> str:
    gpu_names = ", ".join(gpu["name"] for gpu in report["gpus"]) or "none"
    failed = [name for name, value in report["readiness"].items() if not value]
    return "\n".join(
        (
            f"GPUs: {gpu_names}",
            f"PyTorch: {report['torch'].get('version', 'unavailable')}",
            f"CUDA runtime: {report['torch'].get('cuda_runtime', 'unavailable')}",
            f"NCCL: {report['torch'].get('nccl_version', 'unavailable')}",
            f"Strict dual-T4 ready: {report['strict_ready']}",
            f"Unsatisfied checks: {', '.join(failed) if failed else 'none'}",
            f"Session fingerprint: {report['session_fingerprint']}",
        )
    )
