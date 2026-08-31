# Stanford CS25: V2 | Introduction to Transformers w/ Andrej Karpathy

A comprehensive lecture breakdown, notes, architectural analysis, and study guide for **Andrej Karpathy's** guest lecture for Stanford CS25 (*Transformers United*).

---

## 1. Lecture Metadata & Video Reference

* **Course:** Stanford CS25: *Transformers United* (Session V2)
* **Instructor / Speaker:** Andrej Karpathy (ex-Director of AI at Tesla, Founding Member at OpenAI, Founder of Eureka Labs)
* **Host / Host Organization:** Stanford Online & Stanford University
* **Official YouTube Video:** [Stanford CS25: V2 | Introduction to Transformers w/ Andrej Karpathy](https://www.youtube.com/watch?v=XfpMkf4rD6E)
* **Course Website:** [https://web.stanford.edu/class/cs25/](https://web.stanford.edu/class/cs25/)
* **Official Stanford Online Playlist:** [Stanford CS25 Playlist](https://www.youtube.com/playlist?list=PLoROMvodv4rNiJRchCzutFw5ItR_Z27CM)
* **Companion Code / nanoGPT Repo:** [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) & [karpathy/ng-video-lecture](https://github.com/karpathy/ng-video-lecture)
* **Foundational Reading:**
  * Vaswani et al. (2017) — [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762)
  * Radford et al. (2019) — [*Language Models are Unsupervised Multitask Learners (GPT-2)*](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

---

## 2. Executive Lecture Roadmap

```
                               STANFORD CS25 (KARPATHY) ROADMAP
                               
┌─────────────────────────┐     ┌─────────────────────────┐     ┌─────────────────────────┐
│  Historical Evolution   │ ──► │ Attention Core Machine  │ ──► │  Self vs Cross-Attn     │
│  (RNNs ➔ Transformers)  │     │   (Q, K, V Mechanics)   │     │ (Enc-Dec vs Dec-Only)   │
└─────────────────────────┘     └─────────────────────────┘     └─────────────────────────┘
             │                                                               │
             ▼                                                               ▼
┌─────────────────────────┐     ┌─────────────────────────┐     ┌─────────────────────────┐
│ Building nanoGPT / GPT2 │ ──► │ Training Dynamics & SOTA│ ──► │ Multimodal & Future     │
│   (Block-by-Block Code) │     │ (Scaling, AMP, AdamW)   │     │ (Vision, Audio, Latents)│
└─────────────────────────┘     └─────────────────────────┘     └─────────────────────────┘
```

---

## 3. Key Topics & Deep-Dive Notes

### 3.1 Why Transformers Replaced RNNs
* **Sequential Bottleneck:** RNNs/LSTMs process tokens step-by-step ($t_1 \to t_2 \to t_3$). This prevents effective GPU parallelization across time $T$.
* **Information Bottleneck & Vanishing Gradients:** Squeezing an arbitrary-length document into a single fixed-size vector $h_t$ causes loss of early context.
* **Transformers' Breakthrough:**
  * Replaces recurrent loops with pairwise dot-product interactions across all positions in parallel.
  * Constant path length $O(1)$ between any two tokens in sequence length $T$ (improving gradient propagation).

---

### 3.2 The Attention Mechanism: Queries, Keys, and Values
Attention is viewed as a **soft lookup / fuzzy dictionary**:
1. **Query ($Q$):** What a token is searching for.
2. **Key ($K$):** What a token contains / advertises.
3. **Value ($V$):** The actual content / payload transferred if a match occurs.

$$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

* $\sqrt{d_k}$ Scaling factor: Normalizes the variance of the dot products back to $1.0$ when $d_k$ is large, preventing the Softmax from saturating into extreme one-hot distributions with vanishingly small gradients.

---

### 3.3 Transformer Architectures: Taxonomy & Archetypes

```
                        TRANSFORMER ARCHITECTURES
                                    │
         ┌──────────────────────────┼──────────────────────────┐
         ▼                          ▼                          ▼
    ENCODER-ONLY               DECODER-ONLY             ENCODER-DECODER
 (BERT, RoBERTa, etc.)      (GPT-2/3/4, LLaMA, Mistral) (Vaswani 2017, T5, Whisper)
         │                          │                          │
 • Bidirectional Self-Attn  • Causal (Masked) Self-Attn • Bidirectional Enc Attn
 • Every token sees all     • Lower-triangular mask     • Causal Dec Self-Attn
 • Best for: Classification, • Best for: Generation,     • Cross-Attention (Dec ➔ Enc)
   Embeddings, Feature Ext.   Prompting, Reasoning      • Best for: Seq2Seq, Audio, Trans
```

---

### 3.4 Deep Dive: Cross-Attention vs. Self-Attention (The Lecture Slide Breakdown)

As presented in Karpathy's slide from CS25:

```python
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        # ---> To turn this into an Encoder-Decoder Block:
        #      Add one more line here for one more (full) cross-attention!
        #      Calculate QUERY from x, but calculate KEY & VALUE from top of encoder.
        x = x + self.mlp(self.ln_2(x))
        return x
```

#### The Essential Rules of Cross-Attention:
1. **Query Source:** Calculated from $x$ (the decoder's current representation).
2. **Key & Value Source:** Calculated from the encoder's top hidden features ($\text{encoder\_output}$).
3. **No Causal Masking:** Cross-attention attends across the entire source sequence bidirectionally ($T_{\text{enc}}$), because the entire source input (e.g., prompt, source sentence, audio frame sequence) is already fully available.

---

## 4. Lecture Timestamps & Key Moments

| Timestamp | Lecture Topic | Key Takeaway |
|:---|:---|:---|
| **00:00 - 05:00** | Introduction & Historical Context | Language modeling framing, n-grams, and RNN/LSTM background. |
| **05:00 - 18:00** | The Attention Mechanism Explained | Soft associative memory, dot products, scaled dot product attention. |
| **18:00 - 32:00** | Multi-Head Attention & Residual Streams | Splitting heads, linear projections, Pre-LN vs Post-LN stability. |
| **32:00 - 45:00** | Encoder vs Decoder vs Cross-Attention | The original 2017 Transformer Seq2Seq vs Decoder-only GPT simplification. |
| **45:00 - 1:00:00**| Training GPT-2 & Modern LLMs | Data ingestion, BPE tokenization, scaling laws, batch sizes, compute budgets. |
| **1:00:00 - End** | Q&A & Future Frontiers | Multimodal conditioning, context window expansion, efficiency. |

---

## 5. Related Project Materials

* 📖 [cross_attention_vs_self_attention_guide.md](./cross_attention_vs_self_attention_guide.md) — Detailed code, mathematical tensor shapes, and PyTorch module for Cross-Attention.
* 📖 [automatic_mixed_precision_amp_guide.md](./automatic_mixed_precision_amp_guide.md) — Mixed precision training recipe for GPU speedup.
* 📖 [torch_set_float32_matmul_precision_guide.md](./torch_set_float32_matmul_precision_guide.md) — Tensor Core TF32 acceleration guide.
* 📖 [the_unreasonable_effectiveness_of_rnns.md](./the_unreasonable_effectiveness_of_rnns.md) — Karpathy's classic pre-transformer sequence modeling guide.
* 📄 [gpt2_paper.pdf](./gpt2_paper.pdf) — OpenAI GPT-2 Paper.
* 📄 [attention_is_all_you_need.pdf](./attention_is_all_you_need.pdf) — Original 2017 Transformer Paper (Vaswani et al.).
