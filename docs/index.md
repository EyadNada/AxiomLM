# AxiomLM

AxiomLM is a high-performance PyTorch library for modern autoregressive Transformer modeling, custom hardware kernel acceleration, and spectral matrix optimization. Engineered from first principles, it bridges the gap between theoretical deep learning and bare-metal hardware execution, maximizing Model FLOPs Utilization (MFU) on constrained hardware architectures.

[Read Research Paper Draft (PDF) :octicons-file-badge-16:](assets/AxiomLM_Research_Paper_Draft.pdf){ .md-button .md-button--primary }
[View Architecture Whitepaper :octicons-book-16:](report.md){ .md-button }

## Core Capabilities

- **Bare-Metal Efficiency**: Custom fused kernels written in Apple Metal Shading Language (MSL) and ARM NEON C++ SIMD intrinsics, sustaining **1,035.3 tok/s** single-token decode under Q4_G64 quantization and **838.9 tok/s** in speculative verification.
- **Speculative & Quantized Decoding**: In-register 4-bit and 8-bit quantized GEMV (`qmv_q4_g64` / `qmv_q8_g64`), 16-token register-tiled Split-KV Flash-Decode, and batched QMM speculative verification (`verify_step` at **2,538.1 tok/s** compute rate).
- **Spectral Matrix Optimization**: Integration of the Muon optimizer utilizing quintic Newton-Schulz iterations for polar decomposition and 52% training step reduction.
- **Zero-Overhead Inference**: State-cached Key-Value (KV) decode engine achieving O(1) step latency with pre-allocated ring-buffer sliding-window eviction.
- **Modern Architecture Suite**: Native implementations of RoPE, RMSNorm, SwiGLU, and Grouped-Query Attention (3:1 GQA).

For setup instructions and benchmarks, refer to the [GitHub Repository](https://github.com/EyadNada/AxiomLM).
