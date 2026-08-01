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

CommGuard's SDK calibration uses repeated payload groups and an idle-derived
capture threshold before interpreting aggregate rank correlation. A
`partially_supported` result means only the listed payload groups were captured
reliably. NVML PCIe values remain PCIe traffic readings; they are not direct
NCCL byte counts.
