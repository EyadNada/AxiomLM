# Engineering Notes and Research Materials

This directory contains my study notes, mathematical derivations, and technical summaries written while building and optimizing this 124M parameter transformer from scratch.

The goal was to start from the classic 2019 GPT-2 baseline, understand every bottleneck down to the hardware level, and systematically upgrade the engine with modern architectural choices (RoPE, RMSNorm, SwiGLU, GQA) and low-level system optimizations (FlashAttention, BF16 mixed precision, zero-sync gradient accumulation, KV-cache decoding).

---

## Suggested Reading Order

If you are exploring the codebase or trying to understand how the pieces fit together, here is the recommended sequence:

1. **[Modern Technologies & Optimizations Master Guide](file:///Users/apple/Desktop/Projects/gpt-2(124M)/material/modern_llm_technologies_and_optimizations_guide.md)**  
   Start here. A complete end-to-end breakdown comparing classic GPT-2 to modern transformer architectures, including intuitive explanations, mathematical formulations, and benchmarks.

2. **[FlashAttention and Fast Scaled Dot-Product Attention](file:///Users/apple/Desktop/Projects/gpt-2(124M)/material/flash_attention_guide.md)**  
   Explains GPU memory hierarchy (HBM vs. on-chip SRAM), why standard attention stalls on memory bandwidth, and how tiling avoids materializing the full attention matrix in VRAM.

3. **[Key-Value (KV) Cache and Fast Inference](file:///Users/apple/Desktop/Projects/gpt-2(124M)/material/kv_cache_inference_engine_guide.md)**  
   Explains the algorithmic transition from quadratic generation time down to constant per-token latency during autoregression.

4. **[Rotary Position Embeddings (RoPE)](file:///Users/apple/Desktop/Projects/gpt-2(124M)/material/rope_rotary_position_embeddings_guide.md)**  
   Why absolute positional embeddings limit sequence length, and how 2D complex rotations encode relative token distances naturally.

---

## Architecture and Modeling

* **[Rotary Position Embeddings (RoPE)](file:///Users/apple/Desktop/Projects/gpt-2(124M)/material/rope_rotary_position_embeddings_guide.md)**: Complex rotation in 2D coordinate pairs, relative distance encoding, and context length extrapolation.
* **[Root Mean Square Normalization (RMSNorm)](file:///Users/apple/Desktop/Projects/gpt-2(124M)/material/rmsnorm_guide.md)**: Why mean-centering is unnecessary for training stability, and how variance-only scaling saves memory bandwidth.
* **[SwiGLU Activation and Gated FFN](file:///Users/apple/Desktop/Projects/gpt-2(124M)/material/swiglu_activation_guide.md)**: Bilinear gating, dimensional scaling with parameter parity, and loss convergence improvements.
* **[Grouped-Query Attention (GQA)](file:///Users/apple/Desktop/Projects/gpt-2(124M)/material/gqa_grouped_query_attention_guide.md)**: Sharing key-value heads across query head groups to reduce inference cache footprint by 66.7%.
* **[Cross-Attention vs. Self-Attention](file:///Users/apple/Desktop/Projects/gpt-2(124M)/material/cross_attention_vs_self_attention_guide.md)**: Mathematical differences between encoder-decoder attention and decoder-only autoregressive self-attention.
* **[OpenAI GPT-2 Implementation Notes](file:///Users/apple/Desktop/Projects/gpt-2(124M)/material/openai_gpt2_repo_breakdown.md)**: Code-level breakdown of the original 2019 TensorFlow implementation and Karpathy's clean PyTorch recreation.
* **[Karpathy Stanford CS25 Lecture Summary](file:///Users/apple/Desktop/Projects/gpt-2(124M)/material/stanford_cs25_v2_karpathy_transformers.md)**: Takeaways from Andrej Karpathy's transformer mechanics and optimization lecture.

---

## Systems and Hardware Acceleration

* **[FlashAttention and SDPA](file:///Users/apple/Desktop/Projects/gpt-2(124M)/material/flash_attention_guide.md)**: Tiled attention inside SRAM, online softmax, and backward pass recomputation.
* **[Online Softmax Normalizer Calculation](file:///Users/apple/Desktop/Projects/gpt-2(124M)/material/online_normalizer_calculation_for_softmax_guide.md)**: Mathematical derivation of the streaming single-pass softmax trick that makes FlashAttention possible.
* **[Automatic Mixed Precision (AMP)](file:///Users/apple/Desktop/Projects/gpt-2(124M)/material/automatic_mixed_precision_amp_guide.md)**: BF16 and FP16 autocasting, dynamic range differences, and gradient scaling.
* **[Tensor Cores and Mixed Precision](file:///Users/apple/Desktop/Projects/gpt-2(124M)/material/tensor_cores_and_mixed_precision_guide.md)**: Systolic array compute hardware, matrix multiply-accumulate operations, and precision formats.
* **[PyTorch Float32 Matmul Precision (TF32)](file:///Users/apple/Desktop/Projects/gpt-2(124M)/material/torch_set_float32_matmul_precision_guide.md)**: Enabling 19-bit TensorFloat math on Ampere and Hopper GPUs.
* **[PyTorch Compile (torch.compile)](file:///Users/apple/Desktop/Projects/gpt-2(124M)/material/torch_compile_guide.md)**: TorchDynamo graph capture, AOTAutograd, and TorchInductor kernel fusion.

---

## Optimizers and Training

* **[AdamW Optimizer Guide](file:///Users/apple/Desktop/Projects/gpt-2(124M)/material/adamw_optimizer_guide.md)**: Derivation of first and second moment estimators, bias correction, and decoupled weight decay.
* **[Muon Matrix Optimizer Guide](file:///Users/apple/Desktop/Projects/gpt-2(124M)/material/muon_optimizer_guide.md)**: Orthogonal matrix updates using Newton-Schulz iterations for faster neural network convergence.
* **[GPT-3 Training Hyperparameters Guide](file:///Users/apple/Desktop/Projects/gpt-2(124M)/material/gpt3_training_hyperparameters_guide.md)**: Learning rate warmups, cosine schedules, weight decay exclusions, and gradient clipping norms.
* **[Distributed Data Parallel (DDP)](file:///Users/apple/Desktop/Projects/gpt-2(124M)/material/distributed_data_parallel_ddp_guide.md)**: Ring All-Reduce communication patterns, gradient bucketing, and multi-GPU synchronization.

---

## Inference and Generation

* **[Key-Value (KV) Cache Engine](file:///Users/apple/Desktop/Projects/gpt-2(124M)/material/kv_cache_inference_engine_guide.md)**: Memory-mapped key-value state management and decoding performance benchmarks.
* **[Generation and Sampling Strategies](file:///Users/apple/Desktop/Projects/gpt-2(124M)/material/generation_and_sampling_strategies.md)**: Mechanics of greedy search, temperature scaling, top-k filtering, and nucleus (top-p) sampling.

---

## Data Pipeline and Tokenization

* **[TikToken and Byte-Pair Encoding (BPE)](file:///Users/apple/Desktop/Projects/gpt-2(124M)/material/tiktokenizer_guide.md)**: How byte-level BPE constructs the 50,257 vocabulary without out-of-vocabulary tokens.
* **[WebText vs. Modern Pretraining Datasets](file:///Users/apple/Desktop/Projects/gpt-2(124M)/material/datasets_webtext_gpt2_vs_gpt3_guide.md)**: Evolution of training corpora from WebText to Common Crawl, FineWeb, and synthetic datasets like TinyStories.

---

## Primary Papers (Local Copies)

Original source papers collected for reference while writing the code:

* `attention_is_all_you_need.pdf` — Vaswani et al. (2017)
* `gpt2_paper.pdf` — Radford et al. (2019)
* `gpt3_paper.pdf` — Brown et al. (2020)
* `gelu_paper.pdf` — Hendrycks & Gimpel (2016)
* `online_normalizer_calculation_for_softmax_paper.pdf` — Milakov & Gimelshein (2018)
