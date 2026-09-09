import sys

with open("axiomlm/kernels/ops.py", "r") as f:
    content = f.read()

ce_code = """
# ----------------------------------------------------------------------------
# 5. Fused Cross Entropy
# ----------------------------------------------------------------------------

class FusedCrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, targets, ignore_index):
        ctx.save_for_backward(logits, targets)
        ctx.ignore_index = ignore_index
        losses = _METAL_MOD.cross_entropy_forward_mps(logits.contiguous(), targets.contiguous(), ignore_index)
        return losses

    @staticmethod
    def backward(ctx, grad_losses):
        logits, targets = ctx.saved_tensors
        ignore_index = ctx.ignore_index
        grad_logits = _METAL_MOD.cross_entropy_backward_mps(logits.contiguous(), targets.contiguous(), grad_losses.contiguous(), ignore_index)
        return grad_logits, None, None

def fused_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100, reduction: str = 'mean') -> torch.Tensor:
    if _METAL_MOD is not None and logits.device.type == "mps" and logits.dtype == torch.float32:
        orig_shape = logits.shape
        logits_flat = logits.view(-1, orig_shape[-1])
        targets_flat = targets.view(-1)
        
        losses = FusedCrossEntropyFunction.apply(logits_flat, targets_flat, ignore_index)
        
        if reduction == 'mean':
            valid_mask = (targets_flat != ignore_index).float()
            num_valid = valid_mask.sum().clamp(min=1.0)
            return (losses * valid_mask).sum() / num_valid
        elif reduction == 'sum':
            valid_mask = (targets_flat != ignore_index).float()
            return (losses * valid_mask).sum()
        else:
            return losses.view(targets.shape)
            
    return F.cross_entropy(logits, targets, ignore_index=ignore_index, reduction=reduction)
"""

with open("axiomlm/kernels/ops.py", "w") as f:
    f.write(content + "\n" + ce_code)
