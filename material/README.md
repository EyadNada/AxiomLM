# 📚 Axiom-LM Research & Systems Knowledge Library

A comprehensive collection of intuitive, mathematical, and systems-engineering guides covering every modern LLM architecture, hardware acceleration technique, and training optimization used in this repository.

---

## 🌟 Master Reference
* **[🚀 Modern LLM Technologies & Optimizations Master Guide](file:///Users/apple/Desktop/Projects/gpt-2(124M)/material/modern_llm_technologies_and_optimizations_guide.md)** — The complete, visual, all-in-one guide explaining all modern advancements from first principles with diagrams, analogies, and benchmarks.

---

## 🏗️ Modern Architectural Redesigns (LLaMA-3 Spec)
1. **[Rotary Position Embeddings (RoPE)](file:///Users/apple/Desktop/Projects/gpt-2(124M)/material/rope_rotary_position_embeddings_guide.md)** — Complex rotation in 2D coordinate pairs, relative distance encoding, and context length extrapolation.
2. **[Root Mean Square Normalization (RMSNorm)](file:///Users/apple/Desktop/Projects/gpt-2(124M)/material/rmsnorm_guide.md)** — Scaling-only normalization, eliminating mean-centering passes to save ~30% kernel execution time.
3. **[SwiGLU Activation & Gated FFN](file:///Users/apple/Desktop/Projects/gpt-2(124M)/material/swiglu_activation_guide.md)** — Bilinear gated feed-forward networks with $(8/3) d_{\text{model}}$ parameter parity.
4. **[Grouped-Query Attention (GQA)](file:///Users/apple/Desktop/Projects/gpt-2(124M)/material/gqa_grouped_query_attention_guide.md)** — Sharing 4 KV heads across 12 Query heads for a 66.7% reduction in inference VRAM.
5. **[Key-Value (KV) Cache Inference Engine](file:///Users/apple/Desktop/Projects/gpt-2(124M)/material/kv_cache_inference_engine_guide.md)** — $O(1)$ constant-time autoregressive generation eliminating $O(T^2)$ quadratic token recomputation.

---

## ⚡ Systems & Low-Level Hardware Optimizations
1. **[FlashAttention & SDPA Guide](file:///Users/apple/Desktop/Projects/gpt-2(124M)/material/flash_attention_guide.md)** — Tiled on-chip SRAM attention avoiding $O(T^2)$ global VRAM traffic.
2. **[Automatic Mixed Precision (AMP) Guide](file:///Users/apple/Desktop/Projects/gpt-2(124M)/material/automatic_mixed_precision_amp_guide.md)** — BF16 and FP16 autocast forward pass with FP32 master weight updates.
3. **[Tensor Cores & Mixed Precision Guide](file:///Users/apple/Desktop/Projects/gpt-2(124M)/material/tensor_cores_and_mixed_precision_guide.md)** — Hardware execution units, systolic arrays, and FP16/BF16/TF32 formats.
4. **[PyTorch Float32 Matmul Precision (`TF32`)](file:///Users/apple/Desktop/Projects/gpt-2(124M)/material/torch_set_float32_matmul_precision_guide.md)** — Unlocking TensorFloat-32 speedups on NVIDIA Ampere/Hopper architectures.
5. **[PyTorch Compile (`torch.compile`) Guide](file:///Users/apple/Desktop/Projects/gpt-2(124M)/material/torch_compile_guide.md)** — Graph capturing, kernel fusion, and inductor code generation.
6. **[Online Normalizer Calculation for Softmax](file:///Users/apple/Desktop/Projects/gpt-2(124M)/material/online_normalizer_calculation_for_softmax_guide.md)** — Streaming single-pass softmax formulation used inside FlashAttention.

---

## 🔬 Optimizers & Training Science
1. **[AdamW Optimizer Guide](file:///Users/apple/Desktop/Projects/gpt-2(124M)/material/adamw_optimizer_guide.md)** — First and second moment estimators, bias corrections, and decoupled weight decay.
2. **[Muon Matrix Optimizer Guide](file:///Users/apple/Desktop/Projects/gpt-2(124M)/material/muon_optimizer_guide.md)** — Newton-Schulz matrix orthogonalization for faster neural network convergence.
3. **[GPT-3 Training Hyperparameters Guide](file:///Users/apple/Desktop/Projects/gpt-2(124M)/material/gpt3_training_hyperparameters_guide.md)** — Batch scheduling, cosine learning rate decay with linear warmup, and weight decay norms.
4. **[Distributed Data Parallel (DDP) Guide](file:///Users/apple/Desktop/Projects/gpt-2(124M)/material/distributed_data_parallel_ddp_guide.md)** — Ring All-Reduce, gradient synchronization, and multi-GPU scaling.

---

## 📖 Datasets & Tokenization
1. **[TikToken & Byte-Pair Encoding (BPE) Guide](file:///Users/apple/Desktop/Projects/gpt-2(124M)/material/tiktokenizer_guide.md)** — Subword tokenization, regex splitting, and byte-level vocabulary construction.
2. **[WebText vs. Modern Pretraining Datasets Guide](file:///Users/apple/Desktop/Projects/gpt-2(124M)/material/datasets_webtext_gpt2_vs_gpt3_guide.md)** — Dataset curation, deduplication, filtering, and synthetic corpora (TinyStories, FineWeb-Edu).
3. **[Generation & Sampling Strategies Guide](file:///Users/apple/Desktop/Projects/gpt-2(124M)/material/generation_and_sampling_strategies.md)** — Greedy decoding, Temperature scaling, Top-$k$, and Nucleus (Top-$p$) sampling.
