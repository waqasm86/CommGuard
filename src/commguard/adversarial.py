"""Bounded defensive red-team strategy contracts and synchronization semantics."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ParameterBound:
    minimum: float
    maximum: float
    integral: bool = True

    def validate(self, name: str, value: Any) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"adversarial parameter {name!r} must be numeric")
        numeric = float(value)
        if not self.minimum <= numeric <= self.maximum:
            raise ValueError(
                f"adversarial parameter {name!r}={numeric:g} is outside "
                f"[{self.minimum:g}, {self.maximum:g}]"
            )
        if self.integral and numeric != int(numeric):
            raise ValueError(f"adversarial parameter {name!r} must be integral")


@dataclass(frozen=True)
class AdversarialStrategy:
    strategy_id: str
    family: str
    target_label: str
    worker_mode: str
    research_purpose: str
    synchronization_semantics: str
    parameter_bounds: Mapping[str, ParameterBound]
    claim_boundary: str
    enabled_by_default: bool = False
    requires_human_approval: bool = True

    def validate_config(self, config: Mapping[str, Any]) -> None:
        if config.get("strategy_id") != self.strategy_id:
            raise ValueError(
                f"strategy {self.strategy_id!r} config has mismatched strategy_id "
                f"{config.get('strategy_id')!r}"
            )
        if config.get("family") != self.family or config.get("label") != self.target_label:
            raise ValueError(f"strategy {self.strategy_id!r} has mismatched family or label")
        if config.get("mode") != self.worker_mode:
            raise ValueError(f"strategy {self.strategy_id!r} has mismatched worker mode")
        if config.get("designation") != "adversarial":
            raise ValueError(f"strategy {self.strategy_id!r} must be adversarial")
        if config.get("enabled_by_default") is not False:
            raise ValueError(f"strategy {self.strategy_id!r} cannot be enabled by default")
        if config.get("requires_human_approval") is not True:
            raise ValueError(f"strategy {self.strategy_id!r} requires explicit human approval")
        for name, bound in self.parameter_bounds.items():
            if name not in config:
                raise ValueError(f"strategy {self.strategy_id!r} requires parameter {name!r}")
            bound.validate(name, config[name])
        common_bounds = {
            "batch_size": ParameterBound(1, 4),
            "sequence_length": ParameterBound(32, 512),
            "hidden_size": ParameterBound(64, 256),
            "layers": ParameterBound(1, 3),
            "heads": ParameterBound(1, 8),
            "warmup_seconds": ParameterBound(0, 10, integral=False),
        }
        for name, bound in common_bounds.items():
            if name in config:
                bound.validate(name, config[name])
        if (
            "hidden_size" in config
            and "heads" in config
            and int(config["hidden_size"]) % int(config["heads"]) != 0
        ):
            raise ValueError("adversarial hidden_size must be divisible by heads")
        if config.get("precision") not in {None, "float16", "float32", "bfloat16"}:
            raise ValueError("adversarial precision must be float16, float32, or bfloat16")
        measured = float(config.get("min_measured_seconds", 0))
        if self.strategy_id == "segmented_runs":
            if measured != float(config["segment_seconds"]):
                raise ValueError("segmented run duration must equal segment_seconds")
        elif not 35 <= measured <= 60:
            raise ValueError("adversarial measured duration must be within [35, 60] seconds")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


ADVERSARIAL_STRATEGIES: dict[str, AdversarialStrategy] = {
    "gradient_accumulation": AdversarialStrategy(
        strategy_id="gradient_accumulation",
        family="gradient_accumulation",
        target_label="training",
        worker_mode="ddp_train",
        research_purpose="Test a frozen detector against less frequent gradient synchronization.",
        synchronization_semantics=(
            "One DDP synchronization per optimizer step after local microsteps."
        ),
        parameter_bounds={"gradient_accumulation": ParameterBound(2, 16)},
        claim_boundary="A bounded DDP variant; no detector-evasion result exists until executed.",
    ),
    "periodic_local_sgd": AdversarialStrategy(
        strategy_id="periodic_local_sgd",
        family="periodic_local_sgd",
        target_label="training",
        worker_mode="sparse_sync_training",
        research_purpose="Measure detector sensitivity to periodic model averaging.",
        synchronization_semantics="Average model parameters after a fixed number of local steps.",
        parameter_bounds={"local_steps": ParameterBound(2, 100)},
        claim_boundary="Single-host two-rank local SGD, not decentralized multi-node validation.",
    ),
    "diloco_inspired": AdversarialStrategy(
        strategy_id="diloco_inspired",
        family="diloco_inspired",
        target_label="training",
        worker_mode="sparse_sync_training",
        research_purpose="Probe sparse outer synchronization in a T4-sized defensive experiment.",
        synchronization_semantics="Average parameters after bounded inner training steps.",
        parameter_bounds={"inner_steps": ParameterBound(2, 100)},
        claim_boundary=(
            "DiLoCo-inspired parameter averaging only; not a faithful DiLoCo reproduction."
        ),
    ),
    "segmented_runs": AdversarialStrategy(
        strategy_id="segmented_runs",
        family="segmented_runs",
        target_label="training",
        worker_mode="ddp_train",
        research_purpose="Test whether bounded interruption gaps alter communication windows.",
        synchronization_semantics="Ordinary DDP in separately launched bounded process segments.",
        parameter_bounds={
            "segment_seconds": ParameterBound(5, 20),
            "restart_gap_seconds": ParameterBound(1, 5),
            "segment_count": ParameterBound(2, 8),
        },
        claim_boundary=(
            "Local subprocess restarts only; not provider scheduler or checkpoint validation."
        ),
    ),
    "idle_padding": AdversarialStrategy(
        strategy_id="idle_padding",
        family="idle_padding",
        target_label="training",
        worker_mode="ddp_train",
        research_purpose="Measure the cost and detector response of bounded burst/idle shaping.",
        synchronization_semantics="Ordinary DDP synchronization with deterministic idle padding.",
        parameter_bounds={"idle_padding_s": ParameterBound(0.05, 2, integral=False)},
        claim_boundary="Controlled timing perturbation only; not monitoring tampering.",
    ),
    "randomized_synchronization": AdversarialStrategy(
        strategy_id="randomized_synchronization",
        family="randomized_synchronization",
        target_label="training",
        worker_mode="sparse_sync_training",
        research_purpose="Test a deterministic jittered parameter-averaging schedule.",
        synchronization_semantics="Seeded per-step synchronization decisions plus final agreement.",
        parameter_bounds={"sync_probability": ParameterBound(0.1, 0.9, integral=False)},
        claim_boundary="Seeded bounded schedule, not an adaptive attack on a live detector.",
    ),
    "mixed_training_inference": AdversarialStrategy(
        strategy_id="mixed_training_inference",
        family="mixed_training_inference",
        target_label="training",
        worker_mode="ddp_train",
        research_purpose="Measure a frozen detector on alternating training and inference steps.",
        synchronization_semantics="DDP training steps interleaved with forward-only phases.",
        parameter_bounds={"inference_every": ParameterBound(2, 8)},
        claim_boundary="Synthetic phase mixture; no production workload representativeness claim.",
    ),
    "synthetic_communication_decoy": AdversarialStrategy(
        strategy_id="synthetic_communication_decoy",
        family="synthetic_communication_decoy",
        target_label="control",
        worker_mode="synthetic_communication_decoy",
        research_purpose="Measure false positives from bounded non-training collective shapes.",
        synchronization_semantics="Seeded, bounded all-reduce bursts over synthetic tensors.",
        parameter_bounds={
            "payload_mib": ParameterBound(1, 16),
            "burst_collectives": ParameterBound(1, 8),
        },
        claim_boundary="Synthetic hard negative only; no bypass or telemetry manipulation.",
    ),
}


def strategy_for_config(config: Mapping[str, Any]) -> AdversarialStrategy:
    strategy_id = str(config.get("strategy_id", ""))
    try:
        strategy = ADVERSARIAL_STRATEGIES[strategy_id]
    except KeyError as exc:
        raise ValueError(f"unknown adversarial strategy {strategy_id!r}") from exc
    strategy.validate_config(config)
    return strategy


def should_synchronize(strategy_id: str, step: int, config: Mapping[str, Any]) -> bool:
    """Return the shared, rank-independent synchronization decision for a 1-based step."""
    if step < 1:
        raise ValueError("synchronization step must be at least one")
    if strategy_id == "periodic_local_sgd":
        return step % int(config["local_steps"]) == 0
    if strategy_id == "diloco_inspired":
        return step % int(config["inner_steps"]) == 0
    if strategy_id == "randomized_synchronization":
        probability = float(config["sync_probability"])
        seed = int(config.get("seed", 1337))
        return random.Random(f"commguard:{seed}:{step}").random() < probability
    raise ValueError(f"strategy {strategy_id!r} does not use sparse synchronization")


def synchronization_steps(
    strategy_id: str,
    completed_steps: int,
    config: Mapping[str, Any],
    *,
    include_final_agreement: bool = True,
) -> tuple[int, ...]:
    if completed_steps < 0:
        raise ValueError("completed_steps cannot be negative")
    steps = [
        step
        for step in range(1, completed_steps + 1)
        if should_synchronize(strategy_id, step, config)
    ]
    if include_final_agreement and completed_steps and completed_steps not in steps:
        steps.append(completed_steps)
    return tuple(steps)


def simulate_parameter_averaging(
    initial_states: Sequence[float],
    local_updates_by_step: Sequence[Sequence[float]],
    synchronization_schedule: Sequence[int],
) -> tuple[tuple[float, ...], ...]:
    """Dependency-free scalar model of local updates and parameter averaging."""
    if len(initial_states) < 2:
        raise ValueError("simulation requires at least two ranks")
    states = [float(value) for value in initial_states]
    synchronized = set(synchronization_schedule)
    history: list[tuple[float, ...]] = []
    for step, updates in enumerate(local_updates_by_step, start=1):
        if len(updates) != len(states):
            raise ValueError(f"step {step} update count does not match rank count")
        states = [state + float(update) for state, update in zip(states, updates, strict=True)]
        if step in synchronized:
            average = sum(states) / len(states)
            states = [average for _ in states]
        history.append(tuple(states))
    unknown = sorted(synchronized - set(range(1, len(local_updates_by_step) + 1)))
    if unknown:
        raise ValueError(f"synchronization schedule references unknown steps: {unknown}")
    return tuple(history)


@dataclass(frozen=True)
class AdversarialHoldoutPlan:
    development_families: tuple[str, ...]
    hardening_families: tuple[str, ...]
    final_family: str
    final_session_ids: tuple[str, ...]
    final_config_ids: tuple[str, ...]
    seed: int = 20260730
    final_holdout_status: str = "sealed"

    def validate(self) -> None:
        known = set(ADVERSARIAL_STRATEGIES)
        declared = (
            set(self.development_families) | set(self.hardening_families) | {self.final_family}
        )
        if declared != known:
            raise ValueError(
                f"holdout plan must assign every strategy family exactly once: "
                f"missing={sorted(known - declared)} unknown={sorted(declared - known)}"
            )
        if set(self.development_families) & set(self.hardening_families):
            raise ValueError("development and hardening families must be disjoint")
        if self.final_family in self.development_families + self.hardening_families:
            raise ValueError("final family cannot enter development or hardening")
        if not self.final_session_ids or not self.final_config_ids:
            raise ValueError("final session and configuration holdouts must be declared")
        if self.final_holdout_status != "sealed":
            raise ValueError("new adversarial holdout plans must start sealed")

    def round_for(
        self,
        *,
        family: str,
        session_id: str,
        config_id: str,
        release_final: bool = False,
    ) -> str:
        self.validate()
        is_final = self.is_final_identity(
            family=family,
            session_id=session_id,
            config_id=config_id,
        )
        if is_final:
            return "final_test" if release_final else "sealed_final_holdout"
        if family in self.hardening_families:
            return "hardening"
        if family in self.development_families:
            return "development"
        raise ValueError(f"unassigned adversarial family {family!r}")

    def is_final_identity(self, *, family: str, session_id: str, config_id: str) -> bool:
        return (
            family == self.final_family
            or session_id in self.final_session_ids
            or config_id in self.final_config_ids
        )

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)
