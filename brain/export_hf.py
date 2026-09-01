"""
AxiomLM: Automated Hugging Face .safetensors Model Exporter.
"""
import os
import sys
import json
import argparse
import torch
from safetensors.torch import save_file, load_file

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import __main__
from brain.train_gpt2 import GPT, GPTConfig
if not hasattr(__main__, "GPTConfig"):
    setattr(__main__, "GPTConfig", GPTConfig)

def export_checkpoint_to_hf(
    checkpoint_path: str = "checkpoints/model_latest.pt",
    output_dir: str = "exports/AxiomLM-124M",
    model_name: str = "AxiomLM-124M",
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n" + "="*70)
    print(f"📦 AxiomLM: Hugging Face .safetensors Model Exporter")
    print(f"="*70)
    print(f"  • Source Checkpoint : {checkpoint_path}")
    print(f"  • Destination Export : {output_dir}")
    print(f"="*70 + "\n")

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    config = checkpoint.get("config", None)
    step = checkpoint.get("step", 0)

    clean_state_dict = {}
    total_params = 0
    seen_pointers = set()
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor):
            tensor_ptr = v.data_ptr()
            if tensor_ptr in seen_pointers:
                # Clone shared memory tensors (e.g. tied weights lm_head and wte) for safetensors format
                t = v.clone().contiguous()
            else:
                seen_pointers.add(tensor_ptr)
                t = v.contiguous()
            clean_state_dict[k] = t
            total_params += t.numel()

    safetensors_path = os.path.join(output_dir, "model.safetensors")
    save_file(clean_state_dict, safetensors_path, metadata={"format": "pt", "model_name": model_name, "step": str(step)})
    print(f"  ✓ Exported: {safetensors_path} ({total_params:,} parameters, {os.path.getsize(safetensors_path) / 1024 / 1024:.2f} MB)")

    cfg_dict = {
        "architectures": ["AxiomLMForCausalLM"],
        "model_type": "axiomlm",
        "vocab_size": getattr(config, "vocab_size", 50304) if config else 50304,
        "n_positions": getattr(config, "block_size", 1024) if config else 1024,
        "n_embd": getattr(config, "n_embd", 768) if config else 768,
        "n_layer": getattr(config, "n_layer", 12) if config else 12,
        "n_head": getattr(config, "n_head", 12) if config else 12,
        "n_kv_head": getattr(config, "n_kv_head", 4) if config else 4,
        "norm_type": getattr(config, "norm_type", "rmsnorm") if config else "rmsnorm",
        "pos_emb": getattr(config, "pos_emb", "rope") if config else "rope",
        "mlp_type": getattr(config, "mlp_type", "swiglu") if config else "swiglu",
        "rope_theta": getattr(config, "rope_theta", 10000.0) if config else 10000.0,
        "bias": getattr(config, "bias", False) if config else False,
        "bos_token_id": 50256,
        "eos_token_id": 50256,
        "pad_token_id": 50256,
        "initializer_range": 0.02,
        "torch_dtype": "bfloat16",
        "trained_step": step,
    }
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(cfg_dict, f, indent=2)
    print(f"  ✓ Exported: {config_path}")

    gen_dict = {
        "bos_token_id": 50256,
        "eos_token_id": 50256,
        "pad_token_id": 50256,
        "do_sample": True,
        "temperature": 0.8,
        "top_k": 50,
        "top_p": 0.95,
        "repetition_penalty": 1.1,
        "max_length": 256,
    }
    gen_path = os.path.join(output_dir, "generation_config.json")
    with open(gen_path, "w", encoding="utf-8") as f:
        json.dump(gen_dict, f, indent=2)
    print(f"  ✓ Exported: {gen_path}")

    tok_dict = {
        "add_bos_token": False,
        "add_prefix_space": False,
        "bos_token": "<|endoftext|>",
        "clean_up_tokenization_spaces": True,
        "eos_token": "<|endoftext|>",
        "model_max_length": 1024,
        "pad_token": "<|endoftext|>",
        "tokenizer_class": "GPT2TokenizerFast",
        "unk_token": "<|endoftext|>",
    }
    tok_path = os.path.join(output_dir, "tokenizer_config.json")
    with open(tok_path, "w", encoding="utf-8") as f:
        json.dump(tok_dict, f, indent=2)
    print(f"  ✓ Exported: {tok_path}")

    # Attempt to export full tokenizer vocabulary files (vocab.json, merges.txt, tokenizer.json)
    try:
        from transformers import GPT2TokenizerFast
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        tokenizer.save_pretrained(output_dir)
        print(f"  ✓ Exported: Complete Hugging Face tokenizer assets (vocab.json, merges.txt, tokenizer.json)")
    except Exception as e:
        print(f"  [Note] Tokenizer fast export fallback: {e}")

    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(f"""---
language:
- en
- python
license: mit
pipeline_tag: text-generation
tags:
- axiomlm
- llama-3
- muon-optimizer
- triton-kernels
- pytorch
---

# 🧠 {model_name}

Trained **AxiomLM 124M parameter** modern autoregressive Transformer model.

## Architecture Specifications
- **Model Type**: Modern LLaMA-3 Transformer Spec
- **Normalization**: Root Mean Square Normalization (RMSNorm)
- **Position Embeddings**: Rotary Position Embeddings (RoPE, $\\theta=10000.0$)
- **Activation Function**: SwiGLU Gated Activation
- **Attention**: Grouped-Query Attention (GQA, 12 Query Heads, 4 KV Heads)
- **Parameters**: 114M (Active representations with tied embeddings)
- **Trained Steps**: {step}

## Quickstart via AxiomLM Engine

```bash
# Clone and install AxiomLM
git clone https://github.com/EyadNada/AxiomLM.git
cd AxiomLM
pip install -e .

# Run inference with O(1) Key-Value Cache
axiom-generate --checkpoint {output_dir} --prompt "def triton_rmsnorm(x, weight):"
```
""")
    print(f"  ✓ Exported: {readme_path}")

    print(f"\n🎉 Successfully exported Hugging Face artifacts to {output_dir}!")
    return output_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AxiomLM Hugging Face .safetensors Model Exporter")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/model_latest.pt")
    parser.add_argument("--output_dir", type=str, default="exports/AxiomLM-124M")
    parser.add_argument("--model_name", type=str, default="AxiomLM-124M")
    args = parser.parse_args()
    export_checkpoint_to_hf(checkpoint_path=args.checkpoint, output_dir=args.output_dir, model_name=args.model_name)
