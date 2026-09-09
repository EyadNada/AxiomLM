import sys

with open("axiomlm/kernels/ops.py", "r") as f:
    content = f.read()

rope_code = """
# ----------------------------------------------------------------------------
# 4. Fused Rotary Position Embeddings (RoPE) Autograd Function
# ----------------------------------------------------------------------------

class FusedRoPEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, freqs_cis):
        ctx.save_for_backward(freqs_cis)
        if _METAL_MOD is not None and x.device.type == "mps" and x.dtype == torch.float32:
            freqs_real = torch.view_as_real(freqs_cis).contiguous()
            out = _METAL_MOD.apply_rope_mps(x.contiguous(), freqs_real, True)
            return out
        raise RuntimeError("FusedRoPEFunction called without available kernel.")

    @staticmethod
    def backward(ctx, grad_out):
        freqs_cis, = ctx.saved_tensors
        if _METAL_MOD is not None and grad_out.device.type == "mps" and grad_out.dtype == torch.float32:
            freqs_real = torch.view_as_real(freqs_cis).contiguous()
            grad_x = _METAL_MOD.apply_rope_mps(grad_out.contiguous(), freqs_real, False)
            return grad_x, None
        raise RuntimeError("FusedRoPEFunction backward kernel missing for device.")

def fused_apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
    \"\"\"Functional interface for Fused RoPE.\"\"\"
    B, n_head, T, head_dim = x.shape
    freqs_slice = freqs_cis[start_pos : start_pos + T, :]
    
    if _METAL_MOD is not None and x.device.type == "mps" and x.dtype == torch.float32:
        return FusedRoPEFunction.apply(x, freqs_slice)
        
    # Standard PyTorch eager fallback
    orig_dtype = x.dtype
    x_complex = torch.view_as_complex(x.float().reshape(B, n_head, T, -1, 2))
    freqs_slice_expanded = freqs_slice.view(1, 1, T, -1).to(x.device)
    x_rotated = torch.view_as_real(x_complex * freqs_slice_expanded).reshape(B, n_head, T, head_dim)
    return x_rotated.to(orig_dtype)
"""

with open("axiomlm/kernels/ops.py", "w") as f:
    f.write(content + "\n" + rope_code)
