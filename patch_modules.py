import re

with open("axiomlm/models/modules.py", "r") as f:
    content = f.read()

rope_replacement = """def apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
    \"\"\"
    Applies Rotary Position Embeddings (RoPE) to query or key tensors via complex multiplication.
    Args:
        x: Tensor of shape (B, n_head, T, head_dim)
        freqs_cis: Complex tensor of shape (max_seq_len, head_dim // 2)
        start_pos: Offset for KV-cache decoding step
    Returns:
        Rotated tensor of same shape and dtype as x.
    \"\"\"
    if HAS_CUSTOM_KERNELS:
        try:
            from ..kernels.ops import fused_apply_rope
            return fused_apply_rope(x, freqs_cis, start_pos)
        except ImportError:
            pass

    orig_dtype = x.dtype
    B, n_head, T, head_dim = x.shape
    x_complex = torch.view_as_complex(x.float().reshape(B, n_head, T, -1, 2))
    freqs_slice = freqs_cis[start_pos : start_pos + T, :].view(1, 1, T, -1).to(x.device)
    x_rotated = torch.view_as_real(x_complex * freqs_slice).reshape(B, n_head, T, head_dim)
    return x_rotated.to(orig_dtype)"""

content = re.sub(r'def apply_rope.*?return x_rotated\.to\(orig_dtype\)', rope_replacement, content, flags=re.DOTALL)

with open("axiomlm/models/modules.py", "w") as f:
    f.write(content)
