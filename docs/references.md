# References and attribution

These sources inform design choices. Their empirical results are not CommGuard
results and are never copied into the evidence index.

## Research sources

1. Robi Rahman and Sabiha Tajdari, “Detecting Hidden ML Training With
   Zero-Overhead Telemetry,” arXiv:2606.19262v1, 2026.
   <https://arxiv.org/abs/2606.19262>. The supplied PDF motivates the nine NVML
   signals, broad workload/control coverage, whole-run evaluation, and held-out
   adversarial iteration. CommGuard does not reproduce its hardware corpus,
   monitor–evader rounds, or reported scores.
2. Emmanouil Seferis and Tim Fist, “Detecting Compute Structuring in AI
   Governance is likely feasible,” supplied manuscript, 2025. It motivates
   communication/bandwidth as one signal, conservative decisions, segmentation,
   and decentralized/sparse synchronization. Its cloud-provider evidence and
   governance system are outside CommGuard's Kaggle scope.
3. Arthur Douillard, Qixuan Feng, Andrei A. Rusu, Rachita Chhaparia, Yani
   Donchev, Adhiguna Kuncoro, Marc'Aurelio Ranzato, Arthur Szlam, and Jiajun
   Shen, “DiLoCo: Distributed Low-Communication Training of Language Models,”
   arXiv:2311.08105v3, 2024. <https://arxiv.org/abs/2311.08105>. CommGuard's
   bounded two-GPU parameter averaging is only DiLoCo-inspired; it is not a
   faithful algorithmic or scale reproduction.
4. Shaoke Xi et al., “A Few GPUs, A Whole Lotta Scale: Faithful LLM Training
   Emulation with PrismLLM,” arXiv:2605.15617v1, 2026.
   <https://arxiv.org/abs/2605.15617>. It motivates explicit emulation labels and
   warns that ordinary downscaling changes communication, memory, and
   bottlenecks. CommGuard implements no PrismLLM emulation result.
5. Naci Cankaya, “A system overview for near-term, low-trust AI compute
   verification,” version 0.2 working draft, 23 June 2026. The supplied DOCX
   motivates evidence/evaluation separation and layered provenance. Network
   taps, physical security, secure evaluation, replay protocols, and treaty
   architecture are outside this SDK.
6. Mohamed Amine Merzouk, Dmitri Carpov, Mirko Bronzi, Damiano Fornasiere, and
   Adam Oberman, “How Much is Left? LLMs Linearly Encode Their Remaining Output
   Length,” arXiv:2607.05316v1, 2026. <https://arxiv.org/abs/2607.05316>. Its
   hidden-state probing task does not define CommGuard's detector.

The two supplied copies of “How Much is Left?” are byte-identical. The supplied
LessWrong welcome page is community context and supplies no detector method.

## Software and platform references

- NVIDIA Management Library documentation:
  <https://docs.nvidia.com/deploy/nvml-api/>. CommGuard labels the exposed PCIe
  values as NVML PCIe traffic readings, never direct NCCL byte counts.
- NVIDIA Collective Communications Library documentation:
  <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/>.
- PyTorch distributed documentation:
  <https://pytorch.org/docs/stable/distributed.html>.
- Kaggle notebook platform: <https://www.kaggle.com/docs/notebooks>.

Optional NVIDIA tools (`nccl-tests`, `nvbandwidth`, and `cuda-samples`) are not
compiled or downloaded automatically. Their outputs, if later collected, must
remain distinct from NVML measurements and receive their own indexed artifacts.
