# Tiktokenizer & GPT-2 Tokenization Guide

This document provides a guide for using the **Tiktokenizer** visualizer tool and understanding tokenization in **GPT-2 (124M)**.

---

## 1. Web Tool Information

- **URL:** [https://tiktokenizer.vercel.app/?model=gpt2](https://tiktokenizer.vercel.app/?model=gpt2)
- **Model to Select:** `gpt2` (from the model dropdown in the top-right corner)

---

## 2. Input Prompt & Tokenization

### Example 1: `"Hello, I'm a language model."`

#### Input Text
```text
Hello, I'm a language model.
```

#### Token Count
- **Total Tokens:** `8`

#### Token Breakdown Table

| Token Index | Token Piece (Text) | Token ID | Notes |
|:-----------:|:------------------:|:--------:|:------|
| `0` | `Hello` | `15496` | Word start (capitalized) |
| `1` | `,` | `11` | Punctuation (comma) |
| `2` | ` I` | `314` | Leading space + uppercase `I` |
| `3` | `'m` | `1101` | Contraction `'m` |
| `4` | ` a` | `257` | Leading space + word `a` |
| `5` | ` language` | `3303` | Leading space + word `language` |
| `6` | ` model` | `2746` | Leading space + word `model` |
| `7` | `.` | `13` | Punctuation (period / dot) |

#### Expected Output Array
```json
[15496, 11, 314, 1101, 257, 3303, 2746, 13]
```

---

### Example 2: `"Hello, I'm a language model,"` (Used in `train_gpt2.py`)

#### Input Text
```text
Hello, I'm a language model,
```

#### Token Count
- **Total Tokens:** `8`

#### Token Breakdown Table

| Token Index | Token Piece (Text) | Token ID | Notes |
|:-----------:|:------------------:|:--------:|:------|
| `0` | `Hello` | `15496` | Word start |
| `1` | `,` | `11` | Comma |
| `2` | ` I` | `314` | Space + `I` |
| `3` | `'m` | `1101` | Contraction `'m` |
| `4` | ` a` | `257` | Space + `a` |
| `5` | ` language` | `3303` | Space + `language` |
| `6` | ` model` | `2746` | Space + `model` |
| `7` | `,` | `11` | Comma at the end |

#### Expected Output Array
```json
[15496, 11, 314, 1101, 257, 3303, 2746, 11]
```

---

## 3. Python Implementation with `tiktoken`

In your Python code ([train_gpt2.py](file:///Users/apple/Desktop/Projects/gpt-2(124M)/train_gpt2.py)), you can reproduce this exact tokenization using OpenAI's `tiktoken`:

```python
import tiktoken
import torch

# Load the GPT-2 BPE tokenizer
enc = tiktoken.get_encoding('gpt2')

# Encode text into token IDs
prompt = "Hello, I'm a language model,"
tokens = enc.encode(prompt)
print("Token IDs:", tokens)
# Output: [15496, 11, 314, 1101, 257, 3303, 2746, 11]

# Convert tokens to a PyTorch tensor and prepare batch
tokens_tensor = torch.tensor(tokens, dtype=torch.long)              # Shape: (8,)
tokens_batch = tokens_tensor.unsqueeze(0).repeat(5, 1)              # Shape: (5, 8) for 5 sequences
x = tokens_batch.to('mps')                                          # Move to Apple Silicon GPU ('mps') or 'cuda' / 'cpu'
```

---

## 4. Key Takeaways about Byte Pair Encoding (BPE) in GPT-2

1. **Whitespace is part of the token:** In GPT-2's BPE encoding, leading spaces are merged with the following word (e.g. ` language` is token ID `3303`, whereas `language` without a space has a different token ID `16124`).
2. **Punctuation & Contractions:** Words like `I'm` are split into ` I` (`314`) and `'m` (`1101`).
3. **Vocabulary Size:** GPT-2 uses a vocabulary of **50,257** tokens (`vocab_size = 50257`).
