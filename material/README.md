# Technical Research Curriculum & Systems Reference

This directory contains technical guides, mathematical derivations, and systems engineering documentation supporting the AxiomLM architecture, low-level hardware kernels, and training pipelines.

The curriculum covers first-principles Transformer mechanics, hardware memory hierarchies, custom GPU/CPU kernel engineering, spectral optimizer theory (Muon and Newton-Schulz), and inference acceleration.

---

## Suggested Reading Order

For a systematic progression through the systems and mathematical stack:

1. **[Modern LLM Technologies & Optimizations Master Guide](./modern_llm_technologies_and_optimizations_guide.md)**  
   Architectural comparison of baseline Transformer specs vs. modern LLaMA-3/Mistral designs, mathematical formulations, and empirical ablation benchmarks.

2. **[Custom Low-Level Kernels: Triton, Metal MSL & ARM NEON SIMD](./custom_low_level_kernels_triton_metal_neon_guide.md)**  
   Hardware-level kernel fusion, mathematical autograd derivations for fused RMSNorm and SwiGLU, and implementations across OpenAI Triton, Metal Shading Language, and ARM NEON SIMD.

3. **[The Muon Matrix Optimizer Master Guide](./muon_optimizer_guide.md)**  
   Geometric motivation for 2D matrix parameter updates, polar decomposition ($G = U H$), and dual-parameter routing for accelerated step convergence.

4. **[Newton-Schulz Spectral Analysis & Polynomial Approximation Guide](./newton_schulz_spectral_analysis_guide.md)**  
   Minimax quintic polynomial derivation ($p(x) = ax + bx^3 + cx^5$), singular value spectrum flattening, and systolic GEMM efficiency.

5. **[Model FLOPs Utilization (MFU) & Hardware Roofline Guide](./model_flops_utilization_mfu_guide.md)**  
   Analytical compute derivations ($6N$ FLOPs/token), attention quadratic scaling, hardware peak TFLOPs estimation, and Roofline arithmetic intensity boundaries.

6. **[Key-Value (KV) Cache & Inference Engine Guide](./kv_cache_inference_engine_guide.md)**  
   Algorithmic inference scaling, transitioning from quadratic autoregression to constant-time token generation with prefill and decode state caching.

---

## Module 1: Modern Transformer Architecture

* **[Rotary Position Embeddings (RoPE)](./rope_rotary_position_embeddings_guide.md)**: Complex 2D rotations, relative distance preservation, and context window extrapolation.
* **[Root Mean Square Normalization (RMSNorm)](./rmsnorm_guide.md)**: Elimination of mean-centering overhead and variance-only scaling for bandwidth reduction.
* **[SwiGLU Activation & Gated FFN](./swiglu_activation_guide.md)**: Bilinear multiplicative gating, parameter parity dimension scaling ($\frac{8}{3}d$), and gradient flow properties.
* **[Grouped-Query Attention (GQA)](./gqa_grouped_query_attention_guide.md)**: Key-Value head sharing across query groups to reduce inference VRAM footprint by 66.7%.
* **[Cross-Attention vs. Self-Attention](./cross_attention_vs_self_attention_guide.md)**: Structural and mathematical distinctions between encoder-decoder architectures and decoder-only autoregressive models.
* **[OpenAI GPT-2 Implementation Notes](./openai_gpt2_repo_breakdown.md)**: Structural analysis of the original 2019 TensorFlow implementation and clean PyTorch design patterns.
* **[Karpathy Stanford CS25 Lecture Summary](./stanford_cs25_v2_karpathy_transformers.md)**: Core takeaways from transformer mechanics and optimization deep-dives.

---

## Module 2: Hardware Acceleration & Custom Kernels

* **[Custom Low-Level Kernels (Triton / Metal / ARM NEON)](./custom_low_level_kernels_triton_metal_neon_guide.md)**: Fused RMSNorm and SwiGLU operators written in OpenAI Triton (CUDA), Apple Metal MSL, and C++ ARM NEON intrinsics.
* **[FlashAttention & Fast Scaled Dot-Product Attention (SDPA)](./flash_attention_guide.md)**: On-chip SRAM tiling, avoidance of $N \times N$ VRAM materialization, and backward recomputation.
* **[Online Softmax Normalizer Calculation](./online_normalizer_calculation_for_softmax_guide.md)**: Streaming single-pass softmax formulation.
* **[Automatic Mixed Precision (AMP)](./automatic_mixed_precision_amp_guide.md)**: BF16 vs FP16 dynamic ranges, subnormal stability, and gradient scaling mechanics.
* **[Tensor Cores & Mixed-Precision Arithmetic](./tensor_cores_and_mixed_precision_guide.md)**: Systolic array compute hardware, HMMA/MMA instructions, and memory tile alignment.
* **[PyTorch Float32 Matmul Precision (TF32)](./torch_set_float32_matmul_precision_guide.md)**: 19-bit TensorFloat math on NVIDIA Ampere, Ada Lovelace, and Hopper architectures.
* **[PyTorch Compile (`torch.compile`)](./torch_compile_guide.md)**: TorchDynamo graph capture, AOTAutograd graph tracing, and TorchInductor codegen.

---

## Module 3: Optimizers & Training Systems

* **[The Muon Matrix Optimizer Master Guide](./muon_optimizer_guide.md)**: Orthogonal matrix updates via polar decomposition ($G = U H$) and dual-path AdamW/Muon parameter routing.
* **[Newton-Schulz Spectral Analysis Guide](./newton_schulz_spectral_analysis_guide.md)**: Quintic polynomial spectral projector derivation and systolic GEMM optimization.
* **[AdamW Optimizer Guide](./adamw_optimizer_guide.md)**: First and second moment estimators, bias correction schedules, and decoupled weight decay.
* **[GPT-3 Training Hyperparameters Guide](./gpt3_training_hyperparameters_guide.md)**: Warmup ratios, cosine decay schedules, gradient clipping thresholds, and parameter initialization scales.
* **[Distributed Data Parallel (DDP)](./distributed_data_parallel_ddp_guide.md)**: Ring All-Reduce communication topologies, gradient bucketing, and multi-node synchronization.
* **[Scaling Laws & Chinchilla Compute Optimality](./scaling_laws_and_chinchilla_guide.md)**: Kaplan vs. Hoffmann (Chinchilla) scaling frontiers and compute budget allocation.

---

## Module 4: Systems Profiling & Observability

* **[Model FLOPs Utilization (MFU) & Roofline Guide](./model_flops_utilization_mfu_guide.md)**: Analytical FLOPs counting, hardware peak TFLOPs lookup, and arithmetic intensity ceilings.
* **[PyTorch Profiler & Chrome/Perfetto Tracing Guide](./pytorch_profiler_and_chrome_tracing_guide.md)**: Microsecond-accurate timeline profiling, detecting host-device synchronizations, and analyzing Chrome trace graphs.

---

## Module 5: Inference & Data Pipeline

* **[Key-Value (KV) Cache Inference Engine](./kv_cache_inference_engine_guide.md)**: Prefill vs. decode phase state maintenance, memory footprint formulas, and latency benchmarks.
* **[Generation & Sampling Strategies](./generation_and_sampling_strategies.md)**: Greedy decoding, temperature scaling, Top-$k$ truncation, and Top-$p$ (Nucleus) sampling dynamics.
* **[TikToken & Byte-Pair Encoding (BPE)](./tiktokenizer_guide.md)**: Byte-level vocabulary construction, regex splitting rules, and tokenization efficiency.
* **[WebText vs. Modern Systems Datasets](./datasets_webtext_gpt2_vs_gpt3_guide.md)**: Evolution from WebText to FineWeb, RedPajama, Python-Edu, and multi-shard GPU kernel corpora.
* **[AxiomLM Technical Report & Whitepaper](../AxiomLM_Technical_Report.md)**: Formal mathematical formulations, polar Newton-Schulz spectral proofs, and Roofline analysis.

---

## Module 6: Primary Reference Papers

* `attention_is_all_you_need.pdf` — Vaswani et al. (2017)
* `gpt2_paper.pdf` — Radford et al. (2019)
* `gpt3_paper.pdf` — Brown et al. (2020)
* `gelu_paper.pdf` — Hendrycks & Gimpel (2016)
* `online_normalizer_calculation_for_softmax_paper.pdf` — Milakov & Gimelshein (2018)
