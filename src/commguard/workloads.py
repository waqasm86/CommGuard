"""Bounded workload registry and Kaggle-sized benign profiles."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from commguard.adversarial import ADVERSARIAL_STRATEGIES, strategy_for_config

BENIGN_REQUIRED_FAMILIES = (
    "ddp_training",
    "inference_prefill_independent",
    "inference_decode_independent",
    "inference_synchronized",
    "control_compute",
    "control_host_transfer",
    "control_model_or_checkpoint_load",
    "control_idle",
)

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
    "warmup_seconds": 5.0,
    # Five seconds of margin keeps the sampled common interval above the
    # declared 30-second primary window despite asynchronous sampler edges.
    "min_measured_seconds": 35.0,
    "iteration_cap": None,
    "gradient_accumulation": 1,
    "learning_rate": 0.001,
}

WORKLOADS: dict[str, dict[str, Any]] = {
    "calibration_idle": {
        "config_id": "calibration-idle-15s-v1",
        "mode": "idle",
        "label": "calibration",
        "family": "calibration_idle",
        "designation": "calibration",
        "min_measured_seconds": 15.0,
        "iteration_cap": None,
        "idle_interval_s": 0.25,
        "estimated_seconds": 15,
        "warmup_seconds": 0.0,
    },
    "collective_all_reduce_1mib": {
        "config_id": "calibration-all-reduce-1mib-v1",
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
        "warmup_seconds": 0.0,
    },
    "ddp_train": {
        **BASE_MODEL,
        "config_id": "ddp-training-amp-b4-s128-v1",
        "mode": "ddp_train",
        "label": "training",
        "family": "ddp_training",
        "designation": "benign",
        "estimated_seconds": 45,
    },
    "ddp_train_fp32_seq256": {
        **BASE_MODEL,
        "config_id": "ddp-training-fp32-b2-s256-v1",
        "mode": "ddp_train",
        "label": "training",
        "family": "ddp_training",
        "designation": "benign",
        "batch_size": 2,
        "sequence_length": 256,
        "hidden_size": 192,
        "layers": 3,
        "precision": "float32",
        "gradient_scaling": False,
        "estimated_seconds": 50,
    },
    "adversarial_gradient_accumulation": {
        **BASE_MODEL,
        "config_id": "gradient-accumulation-4-v1",
        "strategy_id": "gradient_accumulation",
        "mode": "ddp_train",
        "label": "training",
        "family": "gradient_accumulation",
        "designation": "adversarial",
        "enabled_by_default": False,
        "requires_human_approval": True,
        "gradient_accumulation": 4,
        "estimated_seconds": 90,
    },
    "adversarial_periodic_local_sgd": {
        **BASE_MODEL,
        "config_id": "periodic-local-sgd-steps5-v1",
        "strategy_id": "periodic_local_sgd",
        "mode": "sparse_sync_training",
        "label": "training",
        "family": "periodic_local_sgd",
        "designation": "adversarial",
        "enabled_by_default": False,
        "requires_human_approval": True,
        "local_steps": 5,
        "estimated_seconds": 60,
    },
    "adversarial_diloco_inspired": {
        **BASE_MODEL,
        "config_id": "diloco-inspired-inner10-v1",
        "strategy_id": "diloco_inspired",
        "mode": "sparse_sync_training",
        "label": "training",
        "family": "diloco_inspired",
        "designation": "adversarial",
        "enabled_by_default": False,
        "requires_human_approval": True,
        "inner_steps": 10,
        "estimated_seconds": 60,
    },
    "adversarial_segmented_runs": {
        **BASE_MODEL,
        "config_id": "segmented-runs-4x10s-gap1s-v1",
        "strategy_id": "segmented_runs",
        "mode": "ddp_train",
        "label": "training",
        "family": "segmented_runs",
        "designation": "adversarial",
        "enabled_by_default": False,
        "requires_human_approval": True,
        "segment_seconds": 10,
        "restart_gap_seconds": 1,
        "segment_count": 4,
        "warmup_seconds": 1.0,
        "min_measured_seconds": 10.0,
        "estimated_seconds": 47,
    },
    "adversarial_idle_padding": {
        **BASE_MODEL,
        "config_id": "idle-padding-0.5s-v1",
        "strategy_id": "idle_padding",
        "mode": "ddp_train",
        "label": "training",
        "family": "idle_padding",
        "designation": "adversarial",
        "enabled_by_default": False,
        "requires_human_approval": True,
        "idle_padding_s": 0.5,
        "estimated_seconds": 60,
    },
    "adversarial_randomized_synchronization": {
        **BASE_MODEL,
        "config_id": "randomized-synchronization-p0.25-v1",
        "strategy_id": "randomized_synchronization",
        "mode": "sparse_sync_training",
        "label": "training",
        "family": "randomized_synchronization",
        "designation": "adversarial",
        "enabled_by_default": False,
        "requires_human_approval": True,
        "sync_probability": 0.25,
        "estimated_seconds": 60,
    },
    "adversarial_mixed_training_inference": {
        **BASE_MODEL,
        "config_id": "mixed-training-inference-every2-v1",
        "strategy_id": "mixed_training_inference",
        "mode": "ddp_train",
        "label": "training",
        "family": "mixed_training_inference",
        "designation": "adversarial",
        "enabled_by_default": False,
        "requires_human_approval": True,
        "inference_every": 2,
        "estimated_seconds": 60,
    },
    "adversarial_synthetic_communication_decoy": {
        "config_id": "synthetic-communication-decoy-4mib-burst4-v1",
        "strategy_id": "synthetic_communication_decoy",
        "mode": "synthetic_communication_decoy",
        "label": "control",
        "family": "synthetic_communication_decoy",
        "designation": "adversarial",
        "enabled_by_default": False,
        "requires_human_approval": True,
        "payload_mib": 4,
        "burst_collectives": 4,
        "burst_interval_s": 0.5,
        "iterations": 20,
        "warmup_seconds": 5.0,
        "min_measured_seconds": 35.0,
        "iteration_cap": None,
        "estimated_seconds": 45,
    },
    "inference_prefill_independent": {
        **BASE_MODEL,
        "config_id": "prefill-independent-b4-s128-v1",
        "mode": "inference_independent",
        "label": "inference",
        "family": "inference_prefill_independent",
        "designation": "benign",
        "inference_pattern": "prefill",
        "iterations": 16,
        "estimated_seconds": 40,
    },
    "inference_prefill_independent_seq512": {
        **BASE_MODEL,
        "config_id": "prefill-independent-b1-s512-v1",
        "mode": "inference_independent",
        "label": "inference",
        "family": "inference_prefill_independent",
        "designation": "benign",
        "inference_pattern": "prefill",
        "batch_size": 1,
        "sequence_length": 512,
        "iterations": 16,
        "estimated_seconds": 50,
    },
    "inference_prefill_single_gpu": {
        **BASE_MODEL,
        "config_id": "prefill-single-gpu-b4-s128-v1",
        "mode": "inference_single_gpu",
        "label": "inference",
        "family": "inference_prefill_single_gpu",
        "designation": "benign",
        "inference_pattern": "prefill",
        "iterations": 16,
        "estimated_seconds": 40,
    },
    "inference_decode_independent": {
        **BASE_MODEL,
        "config_id": "decode-independent-b1-context128-v1",
        "mode": "inference_independent",
        "label": "inference",
        "family": "inference_decode_independent",
        "designation": "benign",
        "inference_pattern": "decode",
        "batch_size": 1,
        "iterations": 32,
        "estimated_seconds": 40,
    },
    "inference_decode_independent_batch4": {
        **BASE_MODEL,
        "config_id": "decode-independent-b4-context128-v1",
        "mode": "inference_independent",
        "label": "inference",
        "family": "inference_decode_independent",
        "designation": "benign",
        "inference_pattern": "decode",
        "batch_size": 4,
        "iterations": 32,
        "estimated_seconds": 45,
    },
    "inference_synchronized": {
        **BASE_MODEL,
        "config_id": "prefill-synchronized-every-1-v1",
        "mode": "inference_synchronized",
        "label": "inference",
        "family": "inference_synchronized",
        "designation": "benign",
        "inference_pattern": "prefill",
        "barrier_every": 1,
        "iterations": 16,
        "estimated_seconds": 45,
    },
    "inference_synchronized_every4": {
        **BASE_MODEL,
        "config_id": "prefill-synchronized-every-4-v1",
        "mode": "inference_synchronized",
        "label": "inference",
        "family": "inference_synchronized",
        "designation": "benign",
        "inference_pattern": "prefill",
        "barrier_every": 4,
        "iterations": 16,
        "estimated_seconds": 45,
    },
    "control_compute": {
        "config_id": "control-compute-matmul-2048-v1",
        "mode": "control_compute",
        "label": "control",
        "family": "control_compute",
        "designation": "benign",
        "matrix_size": 2048,
        "iterations": 20,
        "warmup_seconds": 5.0,
        "min_measured_seconds": 35.0,
        "iteration_cap": None,
        "estimated_seconds": 45,
    },
    "control_compute_1024": {
        "config_id": "control-compute-matmul-1024-v1",
        "mode": "control_compute",
        "label": "control",
        "family": "control_compute",
        "designation": "benign",
        "matrix_size": 1024,
        "iterations": 20,
        "warmup_seconds": 5.0,
        "min_measured_seconds": 35.0,
        "iteration_cap": None,
        "estimated_seconds": 45,
    },
    "control_host_transfer": {
        "config_id": "control-host-transfer-64mib-v1",
        "mode": "control_host_transfer",
        "label": "control",
        "family": "control_host_transfer",
        "designation": "benign",
        "payload_mib": 64,
        "iterations": 20,
        "warmup_seconds": 5.0,
        "min_measured_seconds": 35.0,
        "iteration_cap": None,
        "estimated_seconds": 45,
    },
    "control_host_transfer_16mib": {
        "config_id": "control-host-transfer-16mib-v1",
        "mode": "control_host_transfer",
        "label": "control",
        "family": "control_host_transfer",
        "designation": "benign",
        "payload_mib": 16,
        "iterations": 20,
        "warmup_seconds": 5.0,
        "min_measured_seconds": 35.0,
        "iteration_cap": None,
        "estimated_seconds": 45,
    },
    "control_model_load": {
        **BASE_MODEL,
        "config_id": "control-model-checkpoint-load-base-v1",
        "mode": "control_model_load",
        "label": "control",
        "family": "control_model_or_checkpoint_load",
        "designation": "benign",
        "iterations": 8,
        "estimated_seconds": 45,
    },
    "control_peer_copy": {
        "config_id": "control-peer-copy-64mib-v1",
        "mode": "control_peer_copy",
        "label": "control",
        "family": "control_peer_copy",
        "designation": "benign",
        "payload_mib": 64,
        "iterations": 10,
        "warmup_seconds": 5.0,
        "min_measured_seconds": 35.0,
        "iteration_cap": None,
        "estimated_seconds": 45,
        "optional": True,
    },
    "control_idle": {
        "config_id": "control-idle-v1",
        "mode": "control_idle",
        "label": "control",
        "family": "control_idle",
        "designation": "benign",
        "iterations": 20,
        "warmup_seconds": 5.0,
        "min_measured_seconds": 35.0,
        "iteration_cap": None,
        "idle_interval_s": 0.25,
        "estimated_seconds": 45,
    },
}

# ``standard`` is the bounded default pilot. ``extended`` is deliberately
# opt-in and adds parameter variation plus peer copy, which may fail explicitly
# when the runtime topology does not support peer access.
PROFILES = {
    "smoke": [
        "collective_all_reduce_1mib",
        "ddp_train",
        "inference_prefill_independent",
        "control_compute",
    ],
    "standard": [
        "ddp_train",
        "inference_prefill_independent",
        "inference_decode_independent",
        "inference_synchronized",
        "control_compute",
        "control_host_transfer",
        "control_model_load",
        "control_idle",
    ],
}
PROFILES["extended"] = PROFILES["standard"] + [
    "ddp_train_fp32_seq256",
    "inference_prefill_independent_seq512",
    "inference_decode_independent_batch4",
    "inference_synchronized_every4",
    "control_compute_1024",
    "control_host_transfer_16mib",
    "control_peer_copy",
]

ADVERSARIAL_PROFILES = {
    "adversarial_pilot": [
        "adversarial_gradient_accumulation",
        "adversarial_periodic_local_sgd",
        "adversarial_diloco_inspired",
        "adversarial_segmented_runs",
        "adversarial_idle_padding",
        "adversarial_randomized_synchronization",
        "adversarial_mixed_training_inference",
        "adversarial_synthetic_communication_decoy",
    ]
}


def validate_workload_registry() -> None:
    """Validate stable identities and the CPU-inspectable benign execution contract."""
    config_ids: dict[str, str] = {}
    benign_families: set[str] = set()
    for name, config in WORKLOADS.items():
        for field in ("config_id", "mode", "label", "family", "designation"):
            if not config.get(field):
                raise ValueError(f"workload {name!r} requires non-empty {field}")
        config_id = str(config["config_id"])
        if config_id in config_ids:
            raise ValueError(
                f"workloads {config_ids[config_id]!r} and {name!r} share config_id {config_id!r}"
            )
        config_ids[config_id] = name
        if config["designation"] != "benign":
            if config["designation"] == "adversarial":
                strategy_for_config(config)
                minimum_duration = (
                    float(config["segment_seconds"])
                    if config["strategy_id"] == "segmented_runs"
                    else 35.0
                )
                if float(config.get("min_measured_seconds", 0)) < minimum_duration:
                    raise ValueError(
                        f"adversarial workload {name!r} has an invalid measured duration"
                    )
                if float(config.get("estimated_seconds", 0)) < float(
                    config.get("warmup_seconds", 0)
                ) + float(config.get("min_measured_seconds", 0)):
                    raise ValueError(
                        f"adversarial workload {name!r} has an invalid duration estimate"
                    )
            continue
        benign_families.add(str(config["family"]))
        if float(config.get("warmup_seconds", -1)) < 0:
            raise ValueError(f"benign workload {name!r} requires a non-negative warmup")
        if float(config.get("min_measured_seconds", 0)) < 35:
            raise ValueError(f"benign workload {name!r} requires at least 35 measured seconds")
        if float(config.get("estimated_seconds", 0)) < float(
            config.get("warmup_seconds", 0)
        ) + float(config.get("min_measured_seconds", 0)):
            raise ValueError(f"benign workload {name!r} has an invalid duration estimate")
        t4_bounds = {
            "batch_size": 4,
            "sequence_length": 512,
            "hidden_size": 256,
            "layers": 3,
            "matrix_size": 2048,
            "payload_mib": 64,
        }
        for field, maximum in t4_bounds.items():
            if int(config.get(field, 0)) > maximum:
                raise ValueError(
                    f"benign workload {name!r} exceeds T4-safe {field} bound {maximum}"
                )
    missing = set(BENIGN_REQUIRED_FAMILIES) - benign_families
    if missing:
        raise ValueError(
            f"benign workload registry is missing required families: {sorted(missing)}"
        )
    for profile, names in PROFILES.items():
        unknown = set(names) - set(WORKLOADS)
        if unknown:
            raise ValueError(f"profile {profile!r} references unknown workloads: {sorted(unknown)}")
    for profile, names in ADVERSARIAL_PROFILES.items():
        unknown = set(names) - set(WORKLOADS)
        if unknown:
            raise ValueError(f"profile {profile!r} references unknown workloads: {sorted(unknown)}")
        families = {str(WORKLOADS[name]["family"]) for name in names}
        if families != set(ADVERSARIAL_STRATEGIES):
            raise ValueError(
                f"profile {profile!r} must contain every bounded strategy exactly once"
            )


def list_workloads() -> dict[str, dict[str, Any]]:
    """Return a validated defensive copy of all workload configurations."""
    validate_workload_registry()
    return deepcopy(WORKLOADS)


def get_workload(name: str) -> dict[str, Any]:
    validate_workload_registry()
    try:
        return deepcopy(WORKLOADS[name])
    except KeyError as exc:
        raise ValueError(f"unknown workload {name!r}") from exc


def profile_workloads(profile: str) -> list[str]:
    validate_workload_registry()
    try:
        return list(PROFILES[profile])
    except KeyError as exc:
        raise ValueError(f"unknown profile {profile!r}") from exc


def adversarial_profile_workloads(profile: str = "adversarial_pilot") -> list[str]:
    """Return an opt-in strategy profile; this does not authorize execution."""
    validate_workload_registry()
    try:
        return list(ADVERSARIAL_PROFILES[profile])
    except KeyError as exc:
        raise ValueError(f"unknown adversarial profile {profile!r}") from exc
