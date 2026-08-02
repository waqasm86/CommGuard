"""Isolated torchrun worker for calibrations and bounded workloads."""

from __future__ import annotations

import argparse
import json
import os
import random
import socket
import subprocess
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


class EventWriter:
    def __init__(self, path: Path, run_id: str, rank: int, local_rank: int) -> None:
        self.path = path
        self.run_id = run_id
        self.rank = rank
        self.local_rank = local_rank
        path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event: str, **details: Any) -> None:
        record = {
            "artifact_kind": "workload_event",
            "schema_version": "1.0",
            "run_id": self.run_id,
            "rank": self.rank,
            "local_rank": self.local_rank,
            "event": event,
            "wall_time_utc": datetime.now(timezone.utc).isoformat(),
            "monotonic_ns": time.monotonic_ns(),
            "details": details,
        }
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()


def _gpu_uuid(local_rank: int) -> str:
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--id={local_rank}",
            "--query-gpu=uuid",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=5,
    )
    return (
        result.stdout.strip() if result.returncode == 0 else f"unavailable:{result.stderr.strip()}"
    )


def _seed_everything(torch: Any, seed: int, rank: int, deterministic: bool) -> None:
    rank_seed = seed + rank
    random.seed(rank_seed)
    try:
        import numpy as np

        np.random.seed(rank_seed)
    except ImportError:
        pass
    torch.manual_seed(rank_seed)
    torch.cuda.manual_seed_all(rank_seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def _assert_equal_across_ranks(torch: Any, dist: Any, value: Any, label: str) -> list[float]:
    gathered = [torch.zeros_like(value) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, value)
    values = [float(item.item()) for item in gathered]
    if max(values) - min(values) > max(1e-5, abs(values[0]) * 1e-5):
        raise RuntimeError(f"{label} differs across ranks: {values}")
    return values


def _tiny_model(torch: Any, config: dict[str, Any]) -> Any:
    nn = torch.nn

    class TinyLanguageModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            vocab = int(config.get("vocab_size", 2048))
            hidden = int(config.get("hidden_size", 256))
            layers = int(config.get("layers", 2))
            heads = int(config.get("heads", 4))
            self.embedding = nn.Embedding(vocab, hidden)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden,
                nhead=heads,
                dim_feedforward=hidden * 4,
                dropout=0.0,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
            self.output = nn.Linear(hidden, vocab, bias=False)

        def forward(self, tokens: Any) -> Any:
            positions = torch.arange(tokens.shape[1], device=tokens.device)
            causal_mask = torch.triu(
                torch.ones(
                    tokens.shape[1],
                    tokens.shape[1],
                    device=tokens.device,
                    dtype=torch.bool,
                ),
                diagonal=1,
            )
            x = self.embedding(tokens) + 0.01 * positions[None, :, None]
            return self.output(self.encoder(x, mask=causal_mask))

    return TinyLanguageModel()


def _precision_context(torch: Any, precision: str) -> Any:
    if precision == "float16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    if precision == "bfloat16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    from contextlib import nullcontext

    return nullcontext()


def _run_ddp_training(
    torch: Any,
    dist: Any,
    writer: EventWriter,
    config: dict[str, Any],
    local_rank: int,
) -> None:
    nn = torch.nn
    model = _tiny_model(torch, config).cuda(local_rank)
    if config.get("parameter_efficient"):
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.output.parameters():
            parameter.requires_grad = True
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(config.get("learning_rate", 1e-3)),
    )
    iterations = int(config.get("iterations", 8))
    gradient_accumulation = int(config.get("gradient_accumulation", 1))
    batch_size = int(config.get("batch_size", 4))
    sequence_length = int(config.get("sequence_length", 128))
    vocab_size = int(config.get("vocab_size", 2048))
    precision = str(config.get("precision", "float16"))
    gradient_scaling = bool(config.get("gradient_scaling", precision == "float16"))
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=gradient_scaling)
    except AttributeError:
        scaler = torch.cuda.amp.GradScaler(enabled=gradient_scaling)
    idle_padding_s = float(config.get("idle_padding_s", 0.0))
    stagger_s = float(config.get("stagger_rank_s", 0.0)) * writer.rank
    loss_fn = nn.CrossEntropyLoss()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    parameter_bytes = sum(
        parameter.numel() * parameter.element_size() for parameter in model.parameters()
    )
    writer.emit(
        "model_ready",
        parameter_count=parameter_count,
        trainable_parameter_count=trainable_count,
        parameter_bytes=parameter_bytes,
        estimated_adamw_state_bytes=trainable_count * 4 * 2,
        precision=precision,
        gradient_scaling=gradient_scaling,
    )
    dist.barrier()
    for step in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        if stagger_s:
            time.sleep(stagger_s)
        total_loss = 0.0
        for microstep in range(gradient_accumulation):
            tokens = torch.randint(
                vocab_size,
                (batch_size, sequence_length),
                device=local_rank,
            )
            sync = microstep == gradient_accumulation - 1
            context = model.no_sync() if not sync else _precision_context(torch, precision)
            with context:
                if not sync:
                    with _precision_context(torch, precision):
                        logits = model(tokens[:, :-1])
                        loss = loss_fn(logits.reshape(-1, vocab_size), tokens[:, 1:].reshape(-1))
                else:
                    logits = model(tokens[:, :-1])
                    loss = loss_fn(logits.reshape(-1, vocab_size), tokens[:, 1:].reshape(-1))
                loss = loss / gradient_accumulation
            writer.emit("forward_complete", step=step, microstep=microstep)
            scaler.scale(loss).backward()
            total_loss += float(loss.detach())
            writer.emit("backward_complete", step=step, microstep=microstep, synchronized=sync)
        scaler.unscale_(optimizer)
        gradients = [
            parameter.grad.detach().float().sum()
            for parameter in model.parameters()
            if parameter.grad is not None
        ]
        if not gradients:
            raise RuntimeError("no gradients were produced")
        gradient_checksum = torch.stack(gradients).sum()
        checksums = _assert_equal_across_ranks(
            torch, dist, gradient_checksum, "synchronized gradient checksum"
        )
        writer.emit("gradient_sync_complete", step=step, checksums=checksums)
        scaler.step(optimizer)
        scaler.update()
        parameter_checksum = torch.stack(
            [parameter.detach().float().sum() for parameter in model.parameters()]
        ).sum()
        parameter_checksums = _assert_equal_across_ranks(
            torch, dist, parameter_checksum, "optimizer parameter checksum"
        )
        writer.emit(
            "optimizer_step_complete",
            step=step,
            loss=total_loss,
            parameter_checksums=parameter_checksums,
        )
        writer.emit("heartbeat", step=step)
        if idle_padding_s:
            time.sleep(idle_padding_s)
    writer.emit(
        "memory_peak",
        allocated_bytes=int(torch.cuda.max_memory_allocated(local_rank)),
        reserved_bytes=int(torch.cuda.max_memory_reserved(local_rank)),
    )


def _run_inference(
    torch: Any,
    dist: Any,
    writer: EventWriter,
    config: dict[str, Any],
    local_rank: int,
    synchronized: bool,
) -> None:
    model = _tiny_model(torch, config).cuda(local_rank).eval()
    iterations = int(config.get("iterations", 20))
    batch_size = int(config.get("batch_size", 4))
    sequence_length = int(config.get("sequence_length", 128))
    vocab_size = int(config.get("vocab_size", 2048))
    pattern = str(config.get("inference_pattern", "prefill"))
    with torch.inference_mode():
        tokens = torch.randint(vocab_size, (batch_size, sequence_length), device=local_rank)
        for step in range(iterations):
            if synchronized:
                dist.barrier()
            if pattern == "decode":
                logits = model(tokens[:, : min(tokens.shape[1], 32)])
                next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
                tokens = torch.cat((tokens[:, 1:], next_token), dim=1)
            else:
                logits = model(tokens)
            checksum = float(logits.float().sum())
            writer.emit("forward_complete", step=step, checksum=checksum, pattern=pattern)
            writer.emit("heartbeat", step=step)
    writer.emit(
        "memory_peak",
        allocated_bytes=int(torch.cuda.max_memory_allocated(local_rank)),
        reserved_bytes=int(torch.cuda.max_memory_reserved(local_rank)),
    )


def _run_single_gpu_inference(
    torch: Any,
    dist: Any,
    writer: EventWriter,
    config: dict[str, Any],
    local_rank: int,
) -> None:
    if writer.rank == 0:
        _run_inference(torch, dist, writer, config, local_rank, synchronized=False)
        return
    for step in range(int(config.get("iterations", 20))):
        time.sleep(float(config.get("idle_rank_interval_s", 0.05)))
        writer.emit("heartbeat", step=step, role="idle_second_gpu")


def _run_control(
    torch: Any,
    dist: Any,
    writer: EventWriter,
    config: dict[str, Any],
    local_rank: int,
    control: str,
) -> None:
    iterations = int(config.get("iterations", 20))
    if control == "compute":
        size = int(config.get("matrix_size", 2048))
        left = torch.randn(size, size, device=local_rank)
        right = torch.randn(size, size, device=local_rank)
        for step in range(iterations):
            output = left @ right
            writer.emit("cuda_operation_complete", step=step, checksum=float(output[0, 0]))
            writer.emit("heartbeat", step=step)
    elif control == "host_transfer":
        size_mib = int(config.get("payload_mib", 64))
        host = torch.empty(size_mib * 1024 * 1024 // 4, dtype=torch.float32, pin_memory=True)
        for step in range(iterations):
            device = host.to(local_rank, non_blocking=True)
            returned = device.to("cpu", non_blocking=True)
            torch.cuda.synchronize(local_rank)
            writer.emit(
                "cuda_operation_complete",
                step=step,
                direction="host-device-host",
                nominal_bytes=int(host.nbytes * 2),
                checksum=float(returned[0]),
            )
            writer.emit("heartbeat", step=step)
    elif control == "model_load":
        checkpoint_path = writer.path.parent / f"rank-{writer.rank}-checkpoint.pt"
        seed_model = _tiny_model(torch, config)
        torch.save(seed_model.state_dict(), checkpoint_path)
        del seed_model
        for step in range(iterations):
            model = _tiny_model(torch, config)
            state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state)
            model = model.cuda(local_rank)
            checksum = float(next(model.parameters()).float().sum())
            del model
            writer.emit(
                "cuda_operation_complete",
                step=step,
                checksum=checksum,
                checkpoint_bytes=checkpoint_path.stat().st_size,
            )
            writer.emit("heartbeat", step=step)
    elif control == "peer_copy":
        supported = bool(torch.cuda.can_device_access_peer(0, 1))
        if not supported:
            writer.emit(
                "control_unsupported",
                control="peer_copy",
                reason="CUDA peer access from GPU 0 to GPU 1 is unavailable",
            )
            writer.emit("heartbeat", step=0)
        else:
            element_count = int(config.get("payload_mib", 64)) * 1024 * 1024 // 4
            for step in range(iterations):
                dist.barrier()
                if writer.rank == 0:
                    source = torch.ones(element_count, dtype=torch.float32, device="cuda:0")
                    destination = torch.empty(element_count, dtype=torch.float32, device="cuda:1")
                    started = time.perf_counter()
                    destination.copy_(source, non_blocking=True)
                    torch.cuda.synchronize(1)
                    writer.emit(
                        "cuda_operation_complete",
                        step=step,
                        control="peer_copy",
                        nominal_bytes=int(source.nbytes),
                        elapsed_s=time.perf_counter() - started,
                        checksum=float(destination[0]),
                    )
                dist.barrier()
                writer.emit("heartbeat", step=step)
    elif control == "idle":
        for step in range(iterations):
            time.sleep(float(config.get("idle_interval_s", 0.25)))
            writer.emit("heartbeat", step=step)
    else:
        raise ValueError(f"unknown control {control!r}")
    dist.barrier()


def _run_calibration(
    torch: Any,
    dist: Any,
    writer: EventWriter,
    config: dict[str, Any],
    local_rank: int,
) -> None:
    collective = str(config.get("collective", "all_reduce"))
    payload_mib = int(config.get("payload_mib", 1))
    burst_count = int(config.get("iterations", 20))
    collectives_per_burst = int(config.get("burst_iterations", 10))
    burst_interval_s = float(config.get("iteration_interval_s", 0.25))
    dtype = torch.float32
    element_count = payload_mib * 1024 * 1024 // torch.tensor([], dtype=dtype).element_size()
    tensor = torch.full((element_count,), float(writer.rank + 1), dtype=dtype, device=local_rank)
    dist.barrier()
    torch.cuda.synchronize(local_rank)
    nominal_bytes = int(element_count * torch.tensor([], dtype=dtype).element_size())
    writer.emit(
        "collective_start",
        collective=collective,
        payload_mib=payload_mib,
        tensor_bytes=nominal_bytes,
        planned_bursts=burst_count,
        collectives_per_burst=collectives_per_burst,
        burst_interval_s=burst_interval_s,
    )
    started = time.perf_counter()
    for step in range(burst_count):
        burst_started = time.perf_counter()
        for _ in range(collectives_per_burst):
            if collective == "all_reduce":
                tensor.fill_(float(writer.rank + 1))
                dist.all_reduce(tensor)
                expected = 3.0
            elif collective == "broadcast":
                tensor.fill_(float(writer.rank + 1))
                dist.broadcast(tensor, src=0)
                expected = 1.0
            elif collective == "all_gather":
                tensor.fill_(float(writer.rank + 1))
                output = torch.empty(element_count * 2, dtype=dtype, device=local_rank)
                if hasattr(dist, "all_gather_into_tensor"):
                    dist.all_gather_into_tensor(output, tensor)
                else:
                    chunks = list(output.chunk(2))
                    dist.all_gather(chunks, tensor)
                expected = 1.0
                if not (output[0] == 1 and output[-1] == 2):
                    raise RuntimeError("all_gather correctness check failed")
            elif collective == "reduce_scatter":
                output = torch.empty(element_count, dtype=dtype, device=local_rank)
                input_tensor = torch.full(
                    (element_count * 2,),
                    float(writer.rank + 1),
                    dtype=dtype,
                    device=local_rank,
                )
                if not hasattr(dist, "reduce_scatter_tensor"):
                    raise RuntimeError("reduce_scatter_tensor is unavailable")
                dist.reduce_scatter_tensor(output, input_tensor)
                expected = 3.0
                tensor = output
            elif collective == "send_recv":
                tensor.fill_(float(writer.rank + 1))
                if writer.rank == 0:
                    dist.send(tensor, dst=1)
                    dist.recv(tensor, src=1)
                    expected = 2.0
                else:
                    dist.recv(tensor, src=0)
                    tensor.add_(1)
                    dist.send(tensor, dst=0)
                    expected = 2.0
            else:
                raise ValueError(f"unsupported collective {collective!r}")
            if collective != "all_gather" and float(tensor[0]) != expected:
                raise RuntimeError(
                    f"{collective} correctness failed: got {float(tensor[0])}, expected {expected}"
                )
        writer.emit("heartbeat", step=step, collective=collective)
        remaining = burst_interval_s - (time.perf_counter() - burst_started)
        if remaining > 0:
            time.sleep(remaining)
    torch.cuda.synchronize(local_rank)
    elapsed = time.perf_counter() - started
    writer.emit(
        "collective_complete",
        collective=collective,
        payload_mib=payload_mib,
        tensor_bytes=nominal_bytes,
        burst_count=burst_count,
        collectives_per_burst=collectives_per_burst,
        collective_call_count=burst_count * collectives_per_burst,
        elapsed_s=elapsed,
        nominal_payload_throughput_bytes_per_s=(
            nominal_bytes * burst_count * collectives_per_burst / elapsed
        ),
        note="nominal payload and PyTorch timing; not an NVML byte count",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    run_id = str(config["run_id"])
    writer = EventWriter(args.output / f"rank-{rank}.events.jsonl", run_id, rank, local_rank)
    dist = None
    try:
        import torch
        import torch.distributed as dist

        if world_size != 2:
            raise RuntimeError(f"CommGuard requires WORLD_SIZE=2, got {world_size}")
        if torch.cuda.device_count() != 2:
            raise RuntimeError(
                f"CommGuard requires exactly 2 CUDA devices, got {torch.cuda.device_count()}"
            )
        if not dist.is_nccl_available():
            raise RuntimeError("PyTorch NCCL backend is unavailable")
        if local_rank not in (0, 1) or rank not in (0, 1):
            raise RuntimeError(f"unexpected rank mapping rank={rank}, local_rank={local_rank}")
        torch.cuda.set_device(local_rank)
        timeout = timedelta(seconds=float(config.get("process_group_timeout_s", 120)))
        dist.init_process_group(backend="nccl", timeout=timeout)
        if dist.get_backend() != "nccl":
            raise RuntimeError(f"backend fallback rejected: {dist.get_backend()}")
        gpu_uuid = _gpu_uuid(local_rank)
        writer.emit(
            "startup",
            pid=os.getpid(),
            hostname=socket.gethostname(),
            gpu_index=local_rank,
            gpu_uuid=gpu_uuid,
            gpu_name=torch.cuda.get_device_name(local_rank),
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            backend=dist.get_backend(),
        )
        _seed_everything(
            torch,
            int(config.get("seed", 1337)),
            rank,
            bool(config.get("deterministic", False)),
        )
        dist.barrier()
        evidence = torch.tensor([rank + 1], device=local_rank)
        dist.all_reduce(evidence)
        if int(evidence.item()) != 3:
            raise RuntimeError("initial participation reduction failed")
        writer.emit("cuda_operation_complete", checksum=int(evidence.item()))
        mode = str(config["mode"])
        if config.get("inject_rank_crash") == rank:
            raise RuntimeError("injected rank crash")
        if config.get("inject_timeout_rank") == rank:
            time.sleep(float(config.get("process_group_timeout_s", 120)) * 2)
        if mode == "smoke":
            writer.emit("heartbeat", step=0)
        elif mode == "calibration":
            _run_calibration(torch, dist, writer, config, local_rank)
        elif mode == "ddp_train":
            _run_ddp_training(torch, dist, writer, config, local_rank)
        elif mode == "inference_independent":
            _run_inference(torch, dist, writer, config, local_rank, synchronized=False)
        elif mode == "inference_single_gpu":
            _run_single_gpu_inference(torch, dist, writer, config, local_rank)
        elif mode == "inference_synchronized":
            _run_inference(torch, dist, writer, config, local_rank, synchronized=True)
        elif mode.startswith("control_"):
            _run_control(torch, dist, writer, config, local_rank, mode.removeprefix("control_"))
        else:
            raise ValueError(f"unknown mode {mode!r}")
        dist.barrier()
        writer.emit("completion", healthy=True)
        return 0
    except BaseException as exc:
        writer.emit(
            "failure",
            category=type(exc).__name__,
            reason=str(exc),
            traceback=traceback.format_exc(),
        )
        return 1
    finally:
        if dist is not None and dist.is_initialized():
            try:
                dist.destroy_process_group()
                writer.emit("process_group_destroyed")
            except BaseException as exc:
                writer.emit("cleanup_failure", reason=f"{type(exc).__name__}: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
