# Optional AWS multi-node extension

This design is disabled by default and does not block the Kaggle prototype.

1. Validate portability on one `g4dn.xlarge`.
2. Use two single-GPU G4dn nodes in one Availability Zone with direct EC2,
   SSH, PyTorch Distributed, NCCL, and node-local CommGuard collectors.
3. Collect node NIC traffic separately from GPU NVML/PCIe readings.
4. Record instance type, AMI, placement, driver, CUDA, PyTorch, NCCL, ENA/EFA,
   GPU/NIC topology, rank/node identities, and clock synchronization evidence.
5. Only after stable orchestration, consider two `g4dn.12xlarge` nodes for
   four-GPU-per-node EFA-capable tests.
6. Do not introduce EKS initially. Do not launch `g4dn.metal` without a
   budget, schedule, quota confirmation, and explicit approval.
7. Terminate instances immediately after the bounded experiment.

The extension needs a remote process-tree supervisor, unique multi-node
rendezvous, node-local immutable spools, central hash-checked artifact
collection, clock-offset estimation, partial-node failure handling, and distinct
intra-node versus inter-node feature families. T4/G4dn findings must not be
generalized to H100/B200, NVSwitch, RoCE, or frontier training deployments.
