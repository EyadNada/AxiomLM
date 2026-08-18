# Cross-Attention vs. Self-Attention Guide

A comprehensive technical breakdown of **Cross-Attention** (Encoder-Decoder Attention) versus **Self-Attention** in Transformer architectures, based on **Andrej Karpathy's** *Stanford CS25: Introduction to Transformers* lecture and the foundational *Attention Is All You Need* paper (Vaswani et al., 2017).

---

## 1. Overview & Conceptual Intuition

In transformer models, the attention mechanism determines how tokens exchange information:

* **Self-Attention (GPT-2, BERT, LLaMA):** Queries ($Q$), Keys ($K$), and Values ($V$) all originate from the **same sequence** ($x$). Tokens interact with other tokens within their own stream.
* **Cross-Attention (Original Transformer, T5, Whisper, Stable Diffusion):** Queries ($Q$) originate from the **target/decoder sequence** ($x$), while Keys ($K$) and Values ($V$) are derived from an **external source/encoder representation** ($y$ or $\text{encoder\_output}$).

```
┌─────────────────────────────────────────────────────────────────────────┐
│                             SELF-ATTENTION                              │
│                                                                         │
│   Decoder / Sequence Tokens (x) ───┬───► Q (Queries)                    │
│                                    ├───► K (Keys)    ──► Attention(Q,K,V)
│                                    └───► V (Values)                     │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                            CROSS-ATTENTION                              │
│                                                                         │
│   Decoder Stream (x) ──────────────┬───► Q (Queries)                    │
│                                                                         │
│   Encoder Context / Features (y) ──┼───► K (Keys)    ──► Attention(Q,K,V)
│                                    └───► V (Values)                     │
└─────────────────────────────────────────────────────────────────────────┘
```

> **Intuition:**  
> In Cross-Attention, the decoder asks: *"Given what I am currently generating ($Q$), what relevant information should I retrieve from the source context ($K, V$)?"*

---

## 2. Architecture Comparison: Where Cross-Attention Fits

```
              ENCODER                               DECODER (e.g. Translation / T5)
    ┌──────────────────────────┐              ┌──────────────────────────┐
    │     Feed Forward         │              │     Feed Forward         │
    ├──────────────────────────┤              ├──────────────────────────┤
    │  Add & Norm (Residual)   │              │  Add & Norm (Residual)   │
    ├──────────────────────────┤              ├──────────────────────────┤
    │ Bidirectional Self-Attn  │ ──┐          │     CROSS-ATTENTION      │ ◄── [Keys & Values
    ├──────────────────────────┤   │          │ (Q from Dec, K/V from Enc)│     from Encoder]
    │  Add & Norm (Residual)   │   │          ├──────────────────────────┤
    └──────────────────────────┘   │          │  Add & Norm (Residual)   │
                                   └─────────►├──────────────────────────┤
                                              │   Causal Self-Attention  │
                                              ├──────────────────────────┤
                                              │  Add & Norm (Residual)   │
                                              └──────────────────────────┘
```

| Component | Decoder-Only (GPT-2 / LLaMA) | Encoder-Decoder (Vaswani / T5 / Whisper) |
| :--- | :--- | :--- |
| **Layers per Block** | 1 Self-Attention + 1 MLP | 1 Causal Self-Attn + 1 Cross-Attn + 1 MLP |
| **Self-Attention Mask** | Causal (autoregressive lower-triangular) | Causal in Decoder, Bidirectional in Encoder |
| **Cross-Attention Mask** | None (N/A) | **None / Unmasked** (Decoder attends to all encoder tokens) |
| **Use Cases** | Autoregressive language modeling | Translation, Summarization, Audio-to-Text, Multimodal |

---

## 3. Karpathy's CS25 Implementation Breakdown

In decoder-only models like GPT-2, each transformer block contains only **Causal Self-Attention** and an **MLP**. To convert a standard GPT-2 block into an **Encoder-Decoder Block with Cross-Attention**, we inject an extra attention layer into each block.

### A. Standard GPT-2 Block (Self-Attention Only)

```python
import torch
import torch.nn as nn

class GPT2Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        # 1. Causal Self-Attention (x attends to x)
        x = x + self.attn(self.ln_1(x))
        # 2. MLP Feed-Forward
        x = x + self.mlp(self.ln_2(x))
        return x
```

---

### B. Encoder-Decoder Block (With Cross-Attention Added)

```python
class EncoderDecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 1. Causal Self-Attention
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.self_attn = CausalSelfAttention(config)
        
        # 2. Cross-Attention (Conditioned on Encoder Features)
        self.ln_cross = nn.LayerNorm(config.n_embd)
        self.cross_attn = CrossAttention(config)
        
        # 3. Feed-Forward Network
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x, encoder_hidden_states):
        # Step 1: Self-attention over the decoder stream (masked causally)
        x = x + self.self_attn(self.ln_1(x))
        
        # Step 2: Cross-attention -> Q from x, K and V from encoder_hidden_states
        x = x + self.cross_attn(self.ln_cross(x), encoder_hidden_states)
        
        # Step 3: Standard feed-forward MLP
        x = x + self.mlp(self.ln_2(x))
        return x
```

---

## 4. PyTorch Implementation: `CrossAttention` Module

Here is the exact PyTorch implementation showing how $Q$, $K$, and $V$ are computed and combined:

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        # Query projection (applied to the decoder sequence 'x')
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        # Key & Value projections (applied to the encoder hidden states 'enc')
        self.kv_proj = nn.Linear(config.n_embd, 2 * config.n_embd, bias=config.bias)
        
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

    def forward(self, x, encoder_hidden_states, key_padding_mask=None):
        """
        Args:
            x: Decoder hidden states, shape (B, T_dec, C)
            encoder_hidden_states: Encoder output features, shape (B, T_enc, C)
            key_padding_mask: Optional boolean mask (B, T_enc) for encoder padding
        """
        B, T_dec, C = x.size()
        _, T_enc, _ = encoder_hidden_states.size()

        # 1. Project Q from x (Decoder Stream)
        q = self.q_proj(x) # (B, T_dec, C)
        q = q.view(B, T_dec, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, T_dec, hs)

        # 2. Project K and V from encoder_hidden_states (Encoder Stream)
        k, v = self.kv_proj(encoder_hidden_states).split(self.n_embd, dim=2)
        k = k.view(B, T_enc, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, T_enc, hs)
        v = v.view(B, T_enc, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, T_enc, hs)

        # 3. Scaled Dot-Product Attention
        # Note: NO CAUSAL MASK! The decoder is allowed to attend to ALL encoder tokens.
        # FlashAttention / F.scaled_dot_product_attention handles this efficiently:
        y = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=key_padding_mask, 
            is_causal=False # Cross-attention is NEVER causal over the encoder sequence
        ) # (B, nh, T_dec, hs)

        # 4. Re-assemble heads and project back to residual stream
        y = y.transpose(1, 2).contiguous().view(B, T_dec, C)
        y = self.c_proj(y)
        return y
```

---

## 5. Mathematical Dimensions & Tensor Shapes

In Self-Attention, the attention weight matrix is square ($T \times T$). In Cross-Attention, the sequence lengths of the source and target often differ:

| Variable | Description | Shape |
| :--- | :--- | :--- |
| **$x$ (Decoder)** | Current generation sequence | $(B, T_{\text{dec}}, C)$ |
| **$\text{enc}$ (Encoder)** | Source context sequence | $(B, T_{\text{enc}}, C)$ |
| **$Q$** | Projected queries from $x$ | $(B, n_h, T_{\text{dec}}, d_h)$ |
| **$K$** | Projected keys from $\text{enc}$ | $(B, n_h, T_{\text{enc}}, d_h)$ |
| **$V$** | Projected values from $\text{enc}$ | $(B, n_h, T_{\text{enc}}, d_h)$ |
| **$QK^T / \sqrt{d_k}$** | Raw attention affinity scores | $(B, n_h, T_{\text{dec}}, T_{\text{enc}})$ |
| **$\text{Softmax}(QK^T/\sqrt{d_k})$** | Cross-attention probabilities | $(B, n_h, T_{\text{dec}}, T_{\text{enc}})$ |
| **Output $(y)$** | Context-weighted encoder vectors | $(B, T_{\text{dec}}, C)$ |

$$\text{Attention}(Q_{\text{dec}}, K_{\text{enc}}, V_{\text{enc}}) = \text{Softmax}\left( \frac{Q_{\text{dec}} K_{\text{enc}}^T}{\sqrt{d_k}} \right) V_{\text{enc}}$$

---

## 6. Key Distinctions & Rules of Thumb

1. **Query Ownership:**
   * In Self-Attention: $Q, K, V \leftarrow \text{from } x$.
   * In Cross-Attention: $Q \leftarrow \text{from } x$ (Target), $K, V \leftarrow \text{from } \text{context}$ (Source).
2. **Causality & Masking:**
   * **Self-Attention in Decoder:** Must be **causal** (tokens cannot look into future target tokens).
   * **Cross-Attention:** Is **non-causal (full attention)** across the entire encoder context. The entire prompt/audio/image features are known in advance.
3. **Cross-Domain Conditioning:**
   * The encoder and decoder do not even need the same modality or sequence length:
     * **Whisper:** $T_{\text{enc}} = \text{Audio spectrogram frames}$, $T_{\text{dec}} = \text{Text tokens}$.
     * **Stable Diffusion:** $T_{\text{enc}} = \text{CLIP text prompt tokens}$, $T_{\text{dec}} = \text{Flattened image spatial patches/latents}$.

---

## 7. Summary Comparison Cheat Sheet

```
+-----------------------------------------------------------------------------------------------+
| Characteristic        | Self-Attention                     | Cross-Attention                  |
+-----------------------------------------------------------------------------------------------+
| Source of Q           | Decoder hidden states (x)          | Decoder hidden states (x)        |
| Source of K, V        | Decoder hidden states (x)          | Encoder hidden states (enc)      |
| Sequence Lengths      | Always matching (T_dec x T_dec)    | Can differ (T_dec x T_enc)       |
| Masking               | Causal Mask (Lower Triangular)     | Non-Causal (Full bidirectional)  |
| Primary Function      | Modeling autoregressive dependency | Conditioning on external context |
+-----------------------------------------------------------------------------------------------+
```
