# The Unreasonable Effectiveness of Recurrent Neural Networks

- **Author:** Andrej Karpathy
- **Date:** May 21, 2015
- **Original URL:** [https://karpathy.github.io/2015/05/21/rnn-effectiveness/](https://karpathy.github.io/2015/05/21/rnn-effectiveness/)
- **Code Repository (char-rnn):** [https://github.com/karpathy/char-rnn](https://github.com/karpathy/char-rnn)
- **Minimal Python/NumPy Implementation (Gist):** [https://gist.github.com/karpathy/d4dee566867f8291f086](https://gist.github.com/karpathy/d4dee566867f8291f086)

---

## 1. Executive Summary & Core Insights

Before Transformers (GPT, BERT, etc.), **Recurrent Neural Networks (RNNs)** and **Long Short-Term Memory (LSTM)** networks represented the dominant paradigm for sequential modeling and language generation. 

Karpathy's foundational 2015 article demonstrates:
1. **Operating Over Sequences:** How neural networks break free from fixed-size input/output vectors to process variable-length sequential data.
2. **Character-Level Language Modeling:** Predicting the probability distribution of the next character $x_{t+1}$ conditioned on historical context $x_1, \dots, x_t$.
3. **Autoregressive Generation & Temperature Sampling:** Iteratively predicting next tokens, applying temperature scaling to logits, and feeding sampled tokens back as subsequent inputs (the precursor to modern autoregressive LLM decoding).
4. **Interpretable Internal Representations:** Probing hidden states to show that individual RNN/LSTM cells spontaneously learn semantic and syntactic features (e.g. quote matching, indentation counters, code syntax tracking).

---

## 2. Fundamental RNN Mechanics

### Sequence Processing Modes

Traditional feedforward/convolutional networks map fixed-size inputs to fixed-size outputs. Recurrent models generalize computation across variable-length sequences:

```
1. One-to-One:   Vanilla NN (e.g., Image -> Class label)
2. One-to-Many:  Sequence Output (e.g., Image -> Caption text)
3. Many-to-One:  Sequence Input (e.g., Sentiment analysis on review text)
4. Many-to-Many: Asynchronous Seq2Seq (e.g., Machine Translation: English -> French)
5. Many-to-Many: Synchronous Seq2Seq (e.g., Video frame-by-frame labeling)
```

### The Vanilla RNN Update Equation

An RNN maintains an internal hidden state vector $h_t$ that is updated at every time step $t$ given current input $x_t$ and previous hidden state $h_{t-1}$:

$$h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$$
$$y_t = W_{hy} h_t + b_y$$

Where:
- $x_t \in \mathbb{R}^{D}$: Input vector at step $t$ (e.g. 1-hot encoded character or embedding)
- $h_t \in \mathbb{R}^{H}$: Hidden state vector carrying memory across time steps
- $y_t \in \mathbb{R}^{V}$: Output logits over vocabulary size $V$
- $W_{hh}, W_{xh}, W_{hy}$: Learned parameter weight matrices

```python
import numpy as np

class VanillaRNN:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.W_xh = np.random.randn(hidden_dim, input_dim) * 0.01
        self.W_hh = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.W_hy = np.random.randn(output_dim, hidden_dim) * 0.01
        self.h = np.zeros((hidden_dim, 1))

    def step(self, x):
        # Update hidden state
        self.h = np.tanh(np.dot(self.W_hh, self.h) + np.dot(self.W_xh, x))
        # Compute unnormalized log probabilities (logits)
        y = np.dot(self.W_hy, self.h)
        return y
```

---

## 3. Character-Level Language Modeling

### Training Workflow

1. **Vocabulary Construction:** Create a character-level vocabulary $\mathcal{V}$ from text (e.g., characters `[h, e, l, o]`).
2. **1-of-K Encoding:** Represent each character as a one-hot vector $x_t \in \{0, 1\}^{|\mathcal{V}|}$.
3. **Forward Pass:** Compute hidden states $h_t$ and output logits $y_t$ sequentially across sequence length $T$.
4. **Loss Computation:** Apply Softmax + Cross-Entropy loss at each step comparing predicted distribution $p_t = \text{softmax}(y_t)$ against target character $x_{t+1}$:
   $$L_t = -\log p_t(x_{t+1})$$
5. **Backpropagation Through Time (BPTT):** Compute gradients $\frac{\partial L}{\partial W}$ across unrolled time steps and update weights via SGD/Adam/RMSProp.

```
Input:    'h'  ──► [RNN Step] ──► Logits ──► Target: 'e'
           │
Input:    'e'  ──► [RNN Step] ──► Logits ──► Target: 'l'
           │
Input:    'l'  ──► [RNN Step] ──► Logits ──► Target: 'l'
           │
Input:    'l'  ──► [RNN Step] ──► Logits ──► Target: 'o'
```

### Autoregressive Sampling & Temperature

At inference/test time:
1. Feed a seed prompt character (or sequence) into the model.
2. Scale logits with **Temperature** $T$:
   $$p_i = \frac{e^{y_i / T}}{\sum_j e^{y_j / T}}$$
3. Sample the next character $x_{next} \sim p$.
4. Feed $x_{next}$ back into the model as input for the next step.

- **Low Temperature ($T \to 0$):** High confidence, greedy/conservative, repetitive.
- **High Temperature ($T > 1$):** High entropy, diverse/creative, increased typographical errors.

---

## 4. Minimal 100-line Python/NumPy Implementation (`min-char-rnn.py`)

Karpathy's iconic standalone script demonstrating pure character-level RNN training with manual backpropagation and Adagrad:

```python
"""
Minimal character-level Vanilla RNN model. Written by Andrej Karpathy (@karpathy)
BSD License
"""
import numpy as np

# data I/O
data = open('input.txt', 'r').read() # should be simple plain text file
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print(f'data has {data_size} characters, {vocab_size} unique.')
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

# hyperparameters
hidden_size = 100 # size of hidden layer of neurons
seq_length = 25   # number of steps to unroll the RNN for
learning_rate = 1e-1

# model parameters
Wxh = np.random.randn(hidden_size, vocab_size)*0.01 # input to hidden
Whh = np.random.randn(hidden_size, hidden_size)*0.01 # hidden to hidden
Why = np.random.randn(vocab_size, hidden_size)*0.01 # hidden to output
bh = np.zeros((hidden_size, 1))                     # hidden bias
by = np.zeros((vocab_size, 1))                     # output bias

def lossFun(inputs, targets, hprev):
  """
  inputs,targets are both list of integers.
  hprev is Hx1 array of initial hidden state
  returns the loss, gradients on model parameters, and last hidden state
  """
  xs, hs, ys, ps = {}, {}, {}, {}
  hs[-1] = np.copy(hprev)
  loss = 0
  
  # forward pass
  for t in range(len(inputs)):
    xs[t] = np.zeros((vocab_size, 1)) # 1-of-k encoding
    xs[t][inputs[t]] = 1
    hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh) # hidden state
    ys[t] = np.dot(Why, hs[t]) + by                                  # unnormalized log probabilities
    ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))                    # probabilities for next chars
    loss += -np.log(ps[t][targets[t], 0])                            # softmax cross-entropy loss

  # backward pass: compute gradients going backwards
  dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
  dbh, dby = np.zeros_like(bh), np.zeros_like(by)
  dhnext = np.zeros_like(hs[0])
  
  for t in reversed(range(len(inputs))):
    dy = np.copy(ps[t])
    dy[targets[t]] -= 1 # backprop into y
    dWhy += np.dot(dy, hs[t].T)
    dby += dy
    dh = np.dot(Why.T, dy) + dhnext # backprop into h
    dhraw = (1 - hs[t] * hs[t]) * dh # backprop through tanh nonlinearity
    dbh += dhraw
    dWxh += np.dot(dhraw, xs[t].T)
    dWhh += np.dot(dhraw, hs[t-1].T)
    dhnext = np.dot(Whh.T, dhraw)
    
  for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
    np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
    
  return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]

def sample(h, seed_ix, n):
  """ 
  sample a sequence of integers from the model 
  h is memory state, seed_ix is seed letter for first time step
  """
  x = np.zeros((vocab_size, 1))
  x[seed_ix] = 1
  ixes = []
  for t in range(n):
    h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
    y = np.dot(Why, h) + by
    p = np.exp(y) / np.sum(np.exp(y))
    ix = np.random.choice(range(vocab_size), p=p.ravel())
    x = np.zeros((vocab_size, 1))
    x[ix] = 1
    ixes.append(ix)
  return ixes
```

---

## 5. Visualizing Hidden State Neurons

One of the key findings of the post is that despite having no explicit parsing logic or grammar rules, specific hidden neurons spontaneously self-organize to track complex state:

| Discovered Neuron Behavior | Description |
|:---|:---|
| **Quote Detection Cell** | Fires heavily inside double quotes `"` and suppresses outside. |
| **Line-Length / Formatting Cell** | Gradually increases activation until reaching standard ~80-char line limit, then resets upon newline. |
| **If-Statement / Syntax Nesting** | Activates inside C-style blocks and conditional headers. |
| **Comment Depth Tracker** | Changes state inside multi-line or inline comments. |

---

## 6. Historical Lineage: From Char-RNN to GPT-2

| Concept | Karpathy Char-RNN (2015) | GPT-2 (2019 / Our Implementation) |
|:---|:---|:---|
| **Tokenization** | Character-level (1 character = 1 token, vocab ~100) | Byte-Pair Encoding (BPE, byte-level, vocab = 50,257) |
| **Architecture** | Recurrent (RNN / LSTM layers with hidden state $h_t$) | Decoder-Only Transformer (Self-Attention + MLP blocks) |
| **Context Window** | Truncated BPTT (fixed window ~50–100 steps) | Fixed positional embeddings / Context window (1024 tokens) |
| **Parallelization** | Sequential over time steps ($O(T)$ sequential compute) | Parallel computation over all sequence tokens during training |
| **Generation Strategy** | Autoregressive Next-Token + Temperature Sampling | Autoregressive Next-Token + Top-K / Top-P / Temperature |
