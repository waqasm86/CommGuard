"""Bounded workload registry and Kaggle-sized profiles."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

BASE_MODEL = {
    "vocab_size": 2048,
    "hidden_size": 256,
    "layers": 2,
    "heads": 4,
    "batch_size": 4,
    "sequence_length": 128,
    "precision": "float16",
    "gradient_scaling": True,
    "iterations": 8,
    "gradient_accumulation": 1,
    "learning_rate": 0.001,
}

WORKLOADS: dict[str, dict[str, Any]] = {
    "collective_all_reduce_1mib": {
        "mode": "calibration",
        "label": "calibration",
        "family": "nccl_collective",
        "designation": "calibration",
        "collective": "all_reduce",
        "payload_mib": 1,
        "iterations": 20,
        "burst_iterations": 10,
        "iteration_interval_s": 0.25,
        "estimated_seconds": 15,
    },
    "ddp_train": {
        **BASE_MODEL,
        "mode": "ddp_train",
        "label": "training",
        "family": "ddp_full_parameter",
        "designation": "benign",
        "estimated_seconds": 45,
    },
    "ddp_train_grad_accum": {
        **BASE_MODEL,
        "mode": "ddp_train",
        "label": "training",
        "family": "ddp_gradient_accumulation",
        "designation": "adversarial",
        "gradient_accumulation": 4,
        "estimated_seconds": 90,
    },
    "ddp_train_idle_padding": {
        **BASE_MODEL,
        "mode": "ddp_train",
        "label": "training",
        "family": "ddp_idle_padding",
        "designation": "adversarial",
        "idle_padding_s": 0.5,
        "estimated_seconds": 60,
    },
    "ddp_train_parameter_efficient": {
        **BASE_MODEL,
        "mode": "ddp_train",
        "label": "training",
        "family": "parameter_efficient",
        "designation": "adversarial",
        "parameter_efficient": True,
        "estimated_seconds": 45,
    },
    "inference_prefill_independent": {
        **BASE_MODEL,
        "mode": "inference_independent",
        "label": "inference",
        "family": "independent_prefill",
        "designation": "benign",
        "inference_pattern": "prefill",
        "iterations": 16,
        "estimated_seconds": 40,
    },
    "inference_prefill_single_gpu": {
        **BASE_MODEL,
        "mode": "inference_single_gpu",
        "label": "inference",
        "family": "single_gpu_prefill",
        "designation": "benign",
        "inference_pattern": "prefill",
        "iterations": 16,
        "estimated_seconds": 40,
    },
    "inference_decode_independent": {
        **BASE_MODEL,
        "mode": "inference_independent",
        "label": "inference",
        "family": "independent_decode",
        "designation": "benign",
        "inference_pattern": "decode",
        "batch_size": 2,
        "iterations": 32,
        "estimated_seconds": 40,
    },
    "inference_synchronized": {
        **BASE_MODEL,
        "mode": "inference_synchronized",
        "label": "inference",
        "family": "synchronized_prefill",
        "designation": "benign",
        "iterations": 16,
        "estimated_seconds": 45,
    },
    "control_compute": {
        "mode": "control_compute",
        "label": "control",
        "family": "compute_only",
        "designation": "benign",
        "matrix_size": 2048,
        "iterations": 20,
        "estimated_seconds": 30,
    },
    "control_host_transfer": {
        "mode": "control_host_transfer",
        "label": "control",
        "family": "host_device_transfer",
        "designation": "benign",
        "payload_mib": 64,
        "iterations": 20,
        "estimated_seconds": 30,
    },
    "control_model_load": {
        **BASE_MODEL,
        "mode": "control_model_load",
        "label": "control",
        "family": "model_loading",
        "designation": "benign",
        "iterations": 8,
        "estimated_seconds": 30,
    },
    "control_peer_copy": {
        "mode": "control_peer_copy",
        "label": "control",
        "family": "peer_device_copy",
        "designation": "benign",
        "payload_mib": 64,
        "iterations": 10,
        "estimated_seconds": 20,
        "optional": True,
    },
    "control_idle": {
        "mode": "control_idle",
        "label": "control",
        "family": "idle",
        "designation": "benign",
        "iterations": 20,
        "idle_interval_s": 0.25,
        "estimated_seconds": 10,
    },
}

PROFILES = {
    "smoke": [
        "collective_all_reduce_1mib",
        "ddp_train",
        "inference_prefill_single_gpu",
        "inference_prefill_independent",
        "control_compute",
    ],
    "standard": [
        "ddp_train",
        "inference_prefill_single_gpu",
        "inference_prefill_independent",
        "inference_decode_independent",
        "inference_synchronized",
        "control_compute",
        "control_host_transfer",
        "control_model_load",
        "ddp_train_grad_accum",
        "ddp_train_idle_padding",
        "ddp_train_parameter_efficient",
    ],
}
PROFILES["extended"] = PROFILES["standard"] + ["control_peer_copy", "control_idle"]


def list_workloads() -> dict[str, dict[str, Any]]:
    """Return a defensive copy of all workload configurations."""
    return deepcopy(WORKLOADS)


def get_workload(name: str) -> dict[str, Any]:
    try:
        return deepcopy(WORKLOADS[name])
    except KeyError as exc:
        raise ValueError(f"unknown workload {name!r}") from exc


def profile_workloads(profile: str) -> list[str]:
    try:
        return list(PROFILES[profile])
    except KeyError as exc:
        raise ValueError(f"unknown profile {profile!r}") from exc
