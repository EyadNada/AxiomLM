# GPT-3 Training Hyperparameters & Architecture Scaling Guide

This guide documents the model architectures, training hyperparameters, optimization configurations, and data preparation techniques detailed in the landmark paper:
> **"Language Models are Few-Shot Learners"** (Brown et al., OpenAI, 2020) — [arXiv:2005.14165](https://arxiv.org/abs/2005.14165) / [PDF](gpt3_paper.pdf)

While the GPT-2 paper ("Language Models are Unsupervised Multitask Learners", Radford et al., 2019) omitted critical optimization hyperparameter specifics, **GPT-3 Table 2.1 and Appendix B** provided the full standard blueprint used across the modern open-source community to train GPT-2 and GPT-3 architecture scales.

---

## 1. Model Scaling & Hyperparameter Matrix (Table 2.1)

Below is the complete architectural configuration across all 8 model sizes trained in the GPT-3 paper:

| Model Name | Total Params ($n_{\text{params}}$) | Layers ($n_{\text{layers}}$) | Hidden Dimension ($d_{\text{model}}$) | Heads ($n_{\text{heads}}$) | Head Dimension ($d_{\text{head}}$) | Batch Size (Tokens) | Batch Size (Sequences) | Learning Rate ($\eta_{\text{max}}$) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **GPT-3 Small (GPT-2 124M)** | **125M** | **12** | **768** | **12** | **64** | **0.5M (524,288)** | 256 | **$6.0 \times 10^{-4}$ (`6e-4`)** |
| **GPT-3 Medium** | 350M | 24 | 1024 | 16 | 64 | 0.5M | 256 | $3.0 \times 10^{-4}$ (`3e-4`) |
| **GPT-3 Large** | 760M | 24 | 1536 | 16 | 96 | 0.5M | 256 | $2.5 \times 10^{-4}$ (`2.5e-4`) |
| **GPT-3 XL** | 1.3B | 24 | 2048 | 24 | 128 | 1.0M | 512 | $2.0 \times 10^{-4}$ (`2e-4`) |
| **GPT-3 2.7B** | 2.7B | 32 | 2560 | 32 | 80 | 1.0M | 512 | $1.6 \times 10^{-4}$ (`1.6e-4`) |
| **GPT-3 6.7B** | 6.7B | 32 | 4096 | 32 | 128 | 2.0M | 1024 | $1.2 \times 10^{-4}$ (`1.2e-4`) |
| **GPT-3 13B** | 13.0B | 40 | 5120 | 40 | 128 | 2.0M | 1024 | $1.0 \times 10^{-4}$ (`1e-4`) |
| **GPT-3 175B** | 175.0B | 96 | 12288 | 96 | 128 | 3.2M | 1536 | $0.6 \times 10^{-4}$ (`6e-5`) |

> **Key Takeaway for 124M**: The smallest GPT-3 model (`125M`) is architecturally identical to GPT-2 (124M): 12 layers, 768 hidden dimension, 12 attention heads, and $d_{\text{head}} = 64$.

---

## 2. Details of Model Training (Appendix B)

From Appendix B (*Details of Model Training*):

### A. Optimizer Configuration: AdamW
- **Optimizer**: Adam with decoupled weight decay (AdamW).
- **First Moment Momentum ($\beta_1$)**: `0.9`
- **Second Moment Momentum ($\beta_2$)**: `0.95` (Note: default PyTorch AdamW $\beta_2$ is `0.999`. Using `0.95` makes the variance estimate adapt more rapidly to non-stationary gradient dynamics).
- **Numerical Stability ($\epsilon$)**: `1e-8`
- **Weight Decay ($\lambda$)**: `0.1` applied *strictly* to 2D weight matrices (linear projections, embedding weights). Biases and 1D normalization vectors (LayerNorm scales & biases) are exempt from weight decay (`0.0`).

### B. Gradient Clipping
- **Global Gradient Norm**: Clipped at `1.0`.
- Mathematical formulation:
  $$\mathbf{g} \leftarrow \mathbf{g} \cdot \min\left(1, \frac{1.0}{\|\mathbf{g}\|_2}\right)$$
- Protects training from catastrophic exploding gradient updates, especially early in training or when processing outlier token batches.

### C. Learning Rate Schedule: Cosine Decay with Warmup
1. **Linear Warmup**: Over the first **375 million tokens** (or $\sim 1\text{--}2\%$ of total training tokens), the learning rate increases linearly from $0$ up to $\eta_{\text{max}} = 6.0 \times 10^{-4}$.
2. **Cosine Decay**: The learning rate decays following a cosine curve down to **10% of maximum LR** ($\eta_{\text{min}} = 0.1 \times \eta_{\text{max}} = 6.0 \times 10^{-5}$).
3. **Tail Plateau**: Beyond the cosine schedule steps, the learning rate continues indefinitely at the minimum value $\eta_{\text{min}}$.

$$\eta(t) = \begin{cases} 
\frac{t}{T_{\text{warmup}}} \cdot \eta_{\text{max}}, & t < T_{\text{warmup}} \\
\eta_{\text{min}} + \frac{1}{2}\left(1 + \cos\left(\pi \frac{t - T_{\text{warmup}}}{T_{\text{decay}} - T_{\text{warmup}}}\right)\right)(\eta_{\text{max}} - \eta_{\text{min}}), & T_{\text{warmup}} \le t \le T_{\text{decay}} \\
\eta_{\text{min}}, & t > T_{\text{decay}}
\end{cases}$$

### D. Total Batch Size & Gradient Accumulation
- Target Batch Size: **0.5 Million tokens (524,288 tokens)** per forward-backward step.
- In single-GPU training with limited VRAM (e.g., micro-batch size $B=16$, sequence length $T=1024$ $\implies 16,384$ tokens per forward pass), gradient accumulation is used:
  $$\text{grad\_accum\_steps} = \frac{\text{Target Tokens}}{B \times T} = \frac{524,288}{16 \times 1024} = 32 \text{ accumulation steps}$$

### E. Data Sampling & Document Packing
- Documents are concatenated sequentially using `<|endoftext|>` delimiters into unbroken 1024 (or 2048) context windows.
- No special sequence masking is applied across packed documents; the causal mask and the `<|endoftext|>` token enable the model to learn document boundaries naturally.

---

## 3. PyTorch Implementation Reference

Here is how these hyperparameters map directly into PyTorch training code:

```python
import math
import torch

# 1. Hyperparameters
max_lr = 6e-4
min_lr = max_lr * 0.1  # 6e-5
warmup_steps = 715     # e.g., 375M tokens / (524,288 tokens/step) ~ 715 steps
max_steps = 19073      # e.g., 10B tokens / (524,288 tokens/step) ~ 19,073 steps

def get_lr(it: int) -> float:
    # 1) Linear warmup
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    # 2) Constant tail if beyond max steps
    if it > max_steps:
        return min_lr
    # 3) Cosine decay
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0.0 <= decay_ratio <= 1.0
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

# 2. Optimizer Configuration with 2D/1D Parameter Groups
def configure_optimizers(model, weight_decay=0.1, learning_rate=6e-4, device_type='cuda'):
    # Separate parameters into 2D (decayed) and 1D (non-decayed)
    decay_params = [p for n, p in model.named_parameters() if p.dim() >= 2 and p.requires_grad]
    nodecay_params = [p for n, p in model.named_parameters() if p.dim() < 2 and p.requires_grad]
    
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    
    # Fused AdamW on CUDA
    use_fused = (device_type == 'cuda') and ('fused' in torch.optim.AdamW.__init__.__code__.co_varnames)
    optimizer = torch.optim.AdamW(
        optim_groups,
        lr=learning_rate,
        betas=(0.9, 0.95),  # GPT-3 betas
        eps=1e-8,
        fused=use_fused
    )
    return optimizer
```
