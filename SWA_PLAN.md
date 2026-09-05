Plan for SWA:
1. Update `ModelConfig` in `axiomlm/models/transformer.py` to accept `sliding_window: Optional[int]`.
2. Update `CausalSelfAttention.forward` in `axiomlm/models/attention.py`.
   - If using PyTorch SDPA (`F.scaled_dot_product_attention`), we can generate a sliding window boolean mask when `sliding_window` is present, overriding `is_causal=True`.
   - Update `fused_sdpa` signature to accept `sliding_window` and propagate it to Triton kernels.
3. Update `triton_kernels.py`: Add `sliding_window` to `_fused_sdpa_forward_kernel`. Inside the kernel, add the condition `offs_m - offs_n < sliding_window` to the `is_causal` mask.
