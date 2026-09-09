import re

with open("axiomlm/kernels/ops.py", "r") as f:
    content = f.read()

fa_code = """
        if _METAL_MOD is not None and q.device.type == "mps" and q.dtype == torch.float32 and q.size(3) <= 128:
            out = _METAL_MOD.flash_attention_forward_mps(q.contiguous(), k.contiguous(), v.contiguous(), is_causal)
        else:
            if sliding_window is not None and is_causal:
"""

content = content.replace("else:\n            if sliding_window is not None and is_causal:", fa_code)

with open("axiomlm/kernels/ops.py", "w") as f:
    f.write(content)
