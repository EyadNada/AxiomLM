# LLM Scaling Laws & Chinchilla Compute-Optimal Pretraining Guide

A mathematical and empirical guide to **Kaplan Scaling Laws**, **Chinchilla Compute-Optimal Ratios**, **Learning Rate Schedules**, and **Pretraining Budgets** for autoregressive Transformers.

---

## 1. The Foundations: Kaplan et al. (2020) Power Laws

In 2020, OpenAI researchers (Kaplan et al.) demonstrated that cross-entropy validation loss $L$ scales as a power-law with compute budget $C$, dataset size $D$, and parameter count $N$:

$$L(N) = \left( \frac{N_c}{N} \right)^{\alpha_N}, \quad L(D) = \left( \frac{D_c}{D} \right)^{\alpha_D}, \quad L(C) = \left( \frac{C_c}{C} \right)^{\alpha_C}$$

### Key Findings of Kaplan:
* Performance depends strongly on scale ($N, D, C$), while architecture hyperparameters (e.g., depth vs width, number of heads) have minimal impact within reasonable ranges.
* Kaplan originally suggested that as compute increases, parameter count $N$ should scale faster than dataset size $D$ ($N \propto C^{0.73}, D \propto C^{0.27}$).

---

## 2. The Chinchilla Revision: Hoffmann et al. (DeepMind, 2022)

In 2022, DeepMind's Chinchilla paper demonstrated that models from 2020–2021 (including GPT-3 175B and Gopher 280B) were **significantly undertrained** due to sub-optimal compute allocation.

### The Compute-Optimal Rule:
Given a fixed compute budget $C \approx 6 N D$:
$$N \propto C^{0.5}, \quad D \propto C^{0.5}$$

To minimize training loss per FLOP, **parameter count $N$ and token count $D$ should scale in equal proportion**:

$$\frac{D}{N} \approx 20 \text{ tokens per parameter}$$

### Comparison of Model Training Regimes:
| Model | Parameters ($N$) | Training Tokens ($D$) | Ratio ($D/N$) | Status |
| :--- | :---: | :---: | :---: | :--- |
| **GPT-3 (2020)** | 175 Billion | 300 Billion | $1.7\times$ | Severely Undertrained |
| **Chinchilla (2022)** | 70 Billion | 1.4 Trillion | **$20.0\times$** | **Compute-Optimal Baseline** |
| **LLaMA-1 (2023)** | 7 Billion | 1.0 Trillion | $142.8\times$ | Over-trained for Inference Efficiency |
| **LLaMA-3 (2024)** | 8 Billion | 15.0 Trillion | **$1875.0\times$** | Hyper-Trained Frontier |
| **Axiom-LM (124M)** | 124 Million | 20M (Stories) $\rightarrow$ 2.5B (Web) | $20.0\times$ | **Chinchilla Optimal ($2.5\text{B}$ tokens)** |

---

## 3. Learning Rate Schedules & Horizon Tuning

To achieve optimal convergence according to scaling laws:

1. **Warmup Phase:**
   * Linear warmup over the first $1\%$ to $5\%$ of total training steps prevents early gradient explosion.
   $$\eta_t = \eta_{\text{peak}} \times \frac{t}{T_{\text{warmup}}}$$
2. **Cosine Decay Phase:**
   * Decay learning rate according to a cosine curve down to $10\%$ of peak learning rate ($\eta_{\text{min}} = 0.1 \times \eta_{\text{peak}}$):
   $$\eta_t = \eta_{\text{min}} + \frac{1}{2}(\eta_{\text{peak}} - \eta_{\text{min}})\left(1 + \cos\left(\frac{t - T_{\text{warmup}}}{T_{\text{max}} - T_{\text{warmup}}} \pi\right)\right)$$
3. **Horizon Sensitivity:**
   * The cosine horizon must match the total training step budget $T_{\text{max}}$. Decaying too early or too late causes suboptimal perplexity.

---

## 4. Practical Takeaways for Axiom-LM Pretraining

1. **Fast Prototyping (TinyStories - 20M Tokens):**
   * High semantic quality and syntactically rich stories within 4,800 steps ($B=4096$).
2. **Full Pretraining Run (Chinchilla Optimal - 2.5 Billion Tokens):**
   * For general reasoning, train 124M parameters on $2.5\text{B}$ tokens of FineWeb / SlimPajama ($D/N \approx 20$).
