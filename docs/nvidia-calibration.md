# Optional NVIDIA calibration tools

CommGuard has in-SDK PyTorch/NCCL collectives and does not require compilation.
For additional evidence, manually clone and license-review official NVIDIA
repositories in a separate directory. Record exact commit, commands, output,
exit status, driver/runtime, topology, and environment in the artifact tree.

Potential commands after following each project's official build instructions:

```bash
./all_reduce_perf -b 1M -e 1G -f 4 -g 2
./nvbandwidth
./deviceQuery
./bandwidthTest
./simpleP2P
```

Do not automatically install another CUDA runtime or compile NCCL. Do not
equate nccl-tests algorithm bandwidth, bus bandwidth, tensor payload bytes,
PyTorch event timing, or NVML PCIe traffic readings.

CommGuard's full SDK calibration is bounded to five idle repetitions and five
AllReduce repetitions at each of 1, 4, 16, 64, and 128 MiB. Each observation is
explicitly typed as `idle_baseline` or `collective`. The workflow compares 1.0,
0.5, and 0.2 second collector intervals; 0.1 second is diagnostic-only until
its jitter and overhead are measured. A payload repetition is captured when
its PCIe mean exceeds the configured robust idle threshold, which defaults to
the idle median plus three median absolute deviations with an explicit floor.
Each payload needs an 80% capture rate. Monotonicity and dynamic-range checks
remain diagnostics, and the accepted payload sensitivity range is recorded.

`supported` means the required accepted payload range met its
repetition/capture and participation gates. `partially_supported` names a
reliable subset and remains distinct from `inconclusive`, which means the
available evidence cannot decide support. `not_supported` records a valid
negative capability result; `failed` records an execution or integrity
failure. Missing idle evidence or fewer than five usable repetitions cannot
support full-protocol evidence.

Schema-1 results remain readable only through explicitly marked legacy
compatibility. That label never means the modern idle-aware gate passed and
does not reinterpret historical evidence. NVML PCIe values remain PCIe traffic
readings; they are not direct NCCL byte counts.
