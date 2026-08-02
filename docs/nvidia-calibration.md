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

CommGuard's standard SDK calibration is bounded to three idle repetitions and
three AllReduce repetitions at each of 1, 4, 16, and 64 MiB. Each observation is
explicitly typed as `idle_baseline` or `collective`. A payload repetition is
captured only when its PCIe mean exceeds both three times the idle median and
the idle median plus 1 MB/s; the effective threshold is the larger value. Each
payload needs an 80% capture rate, the payload medians need Spearman rank
correlation of at least 0.7, and their dynamic range must be at least 1.2.

`supported` means every requested payload group met its repetition/capture gate
and all aggregate gates passed. `partially_supported` is inconclusive and names
the reliable and unreliable payload groups. `not_supported` records a hard
failure such as missing/unusable idle evidence, unsupported PCIe readings, no
reliably captured payload, or failed monotonicity/dynamic-range gates. Missing
idle observations or fewer than three usable repetitions can never support new
schema-2 evidence.

Schema-1 results remain readable only through explicitly marked legacy
compatibility. That label never means the modern idle-aware gate passed and
does not reinterpret historical evidence. NVML PCIe values remain PCIe traffic
readings; they are not direct NCCL byte counts.
