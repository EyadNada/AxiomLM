# AxiomLM

AxiomLM is a high-performance PyTorch library for modern autoregressive Transformer modeling, custom hardware kernel acceleration, and spectral matrix optimization. Engineered from first principles, it bridges the gap between theoretical deep learning and bare-metal hardware execution, maximizing Model FLOPs Utilization (MFU) on constrained hardware architectures.

[Read Research Paper Draft (PDF) :octicons-file-badge-16:](assets/AxiomLM_Research_Paper_Draft.pdf){ .md-button .md-button--primary }
[View Architecture Whitepaper :octicons-book-16:](report.md){ .md-button }

## Core Capabilities

- **Bare-Metal Efficiency**: Custom fused kernels written in Apple Metal Shading Language (MSL) and ARM NEON C++ SIMD intrinsics.
- **Spectral Matrix Optimization**: Integration of the Muon optimizer utilizing quintic Newton-Schulz iterations.
- **Zero-Overhead Inference**: State-cached Key-Value (KV) decode engine achieving O(1) step latency.
- **Modern Architecture Suite**: Native implementations of RoPE, RMSNorm, SwiGLU, and Grouped-Query Attention.

For setup instructions and benchmarks, refer to the [GitHub Repository](https://github.com/EyadNada/AxiomLM).
