"""
AxiomLM Muon (MomentUm Orthogonalized by Newton-Schulz) Matrix Optimizer.
"""
from typing import List, Optional, Any
import torch
import torch.nn as nn


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """
    Computes the polar orthogonalization factor of 2D matrix G using 5-step quintic Newton-Schulz iteration.
    Transforms matrix singular values toward sigma_i = 1.0.
    """
    assert len(G.shape) == 2, "G must be a 2D tensor"
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16() if G.dtype in [torch.float32, torch.bfloat16] else G.float()
    X = X / (X.norm() + eps)
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-Schulz optimizer.
    Optimizes 2D parameter matrices by replacing gradient coordinates with orthogonal updates.
    """
    def __init__(self, params: Any, lr: float = 0.02, momentum: float = 0.95, nesterov: bool = True, ns_steps: int = 5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Any] = None) -> Optional[float]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']
            ns_steps = group['ns_steps']

            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                if g.ndim != 2:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(g)

                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)

                if nesterov:
                    update = g + momentum * buf
                else:
                    update = buf

                # Polar orthogonalization via 5-step Newton-Schulz
                update_ortho = zeropower_via_newtonschulz5(update, steps=ns_steps)
                
                # Scale update by aspect ratio heuristic
                scale = max(1.0, p.size(0) / p.size(1)) ** 0.5
                p.data.add_(update_ortho, alpha=-lr * scale)

        return loss
