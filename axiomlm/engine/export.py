"""
AxiomLM Hugging Face Safetensors Model Exporter.
"""
import os
import sys
import json
import argparse
from typing import Optional, Dict, Any
import torch

from ..models.transformer import Transformer, ModelConfig, GPT, GPTConfig


def export_checkpoint_to_hf(
    checkpoint_path: str,
    output_dir: str,
    model_name: str = "AxiomLM-124M-Modern",
    license_type: str = "mit",
) -> None:
    """
    Loads an AxiomLM PyTorch .pt checkpoint, converts its tensors to safetensors format,
    and exports config.json, generation_config.json, tokenizer_config.json, vocab.json,
    merges.txt, tokenizer.json, and a model card.
    """
    try:
        from safetensors.torch import save_file
    except ImportError:
        raise ImportError("Package 'safetensors' is required. Install via `pip install safetensors`.")

    try:
        from transformers import GPT2TokenizerFast
    except ImportError:
        GPT2TokenizerFast = None  # type: ignore

    os.makedirs(output_dir, exist_ok=True)
    print("=" * 70)
    print("📦 AxiomLM: Hugging Face .safetensors Model Exporter")
    print("=" * 70)
    print(f"  • Source Checkpoint : {checkpoint_path}")
    print(f"  • Destination Export : {output_dir}")
    print("=" * 70)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if "config" in checkpoint:
        cfg: ModelConfig = checkpoint["config"]
    else:
        print("  ⚠️  Config not found in checkpoint dict. Using default modern specification.")
        cfg = ModelConfig(
            block_size=1024,
            vocab_size=50304,
            n_layer=12,
            n_head=12,
            n_embd=768,
            n_kv_head=4,
            norm_type="rmsnorm",
            pos_emb="rope",
            mlp_type="swiglu",
            bias=False,
        )

    raw_state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    cleaned_state_dict: Dict[str, torch.Tensor] = {}

    for k, v in raw_state_dict.items():
        clean_key = k.replace("_orig_mod.", "").replace("module.", "")
        if not clean_key.endswith(".attn.bias") and not clean_key.endswith(".freqs_cis"):
            cleaned_state_dict[clean_key] = v.contiguous()

    # 1. Export model.safetensors
    safetensors_path = os.path.join(output_dir, "model.safetensors")
    save_file(cleaned_state_dict, safetensors_path)
    total_params = sum(p.numel() for p in cleaned_state_dict.values())
    filesize_mb = os.path.getsize(safetensors_path) / (1024 * 1024)
    print(f"\n  ✓ Exported: {safetensors_path} ({total_params:,} parameters, {filesize_mb:.2f} MB)")

    # 2. Export config.json
    hf_config = {
        "architectures": ["AxiomLMForCausalLM"],
        "model_type": "axiomlm",
        "vocab_size": cfg.vocab_size,
        "hidden_size": cfg.n_embd,
        "num_hidden_layers": cfg.n_layer,
        "num_attention_heads": cfg.n_head,
        "num_key_value_heads": cfg.n_kv_head if cfg.n_kv_head is not None else cfg.n_head,
        "intermediate_size": int(2 * (4 * cfg.n_embd) / 3),
        "max_position_embeddings": cfg.block_size,
        "rms_norm_eps": 1e-6 if cfg.norm_type == "rmsnorm" else 1e-5,
        "norm_type": cfg.norm_type,
        "pos_emb": cfg.pos_emb,
        "mlp_type": cfg.mlp_type,
        "rope_theta": getattr(cfg, "rope_theta", 10000.0),
        "torch_dtype": "float32",
        "transformers_version": "4.44.0",
    }
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(hf_config, f, indent=2)
    print(f"  ✓ Exported: {config_path}")

    # 3. Export generation_config.json
    gen_config = {
        "bos_token_id": 50256,
        "eos_token_id": 50256,
        "pad_token_id": 50256,
        "max_length": cfg.block_size,
        "do_sample": True,
        "temperature": 0.8,
        "top_p": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.1,
    }
    gen_config_path = os.path.join(output_dir, "generation_config.json")
    with open(gen_config_path, "w", encoding="utf-8") as f:
        json.dump(gen_config, f, indent=2)
    print(f"  ✓ Exported: {gen_config_path}")

    # 4. Export Tokenizer Configuration & Files
    tok_config = {
        "add_prefix_space": False,
        "bos_token": "<|endoftext|>",
        "eos_token": "<|endoftext|>",
        "unk_token": "<|endoftext|>",
        "pad_token": "<|endoftext|>",
        "model_max_length": cfg.block_size,
        "tokenizer_class": "GPT2Tokenizer",
    }
    tok_config_path = os.path.join(output_dir, "tokenizer_config.json")
    with open(tok_config_path, "w", encoding="utf-8") as f:
        json.dump(tok_config, f, indent=2)
    print(f"  ✓ Exported: {tok_config_path}")

    if GPT2TokenizerFast is not None:
        try:
            tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
            tokenizer.save_pretrained(output_dir)
            print("  ✓ Exported: Complete Hugging Face tokenizer assets (vocab.json, merges.txt, tokenizer.json)")
        except Exception as e:
            print(f"  ⚠️  Notice: Tokenizer files could not be saved automatically: {e}")

    # 5. Export README.md (Model Card)
    model_card = f"""---
language:
- en
- python
tags:
- axiomlm
- llama-3
- muon-optimizer
- triton-kernels
- pytorch
license: {license_type}
---

# {model_name}

AxiomLM Modern 124M autoregressive language model trained using the **Muon (5-step Newton-Schulz) optimizer**, **LLaMA-3 architectural specifications (RoPE + RMSNorm + SwiGLU + GQA)**, and bare-metal fused kernels.

## Model Specifications
* **Parameters**: ~{total_params / 1e6:.1f}M
* **Layers**: {cfg.n_layer}
* **Hidden Size**: {cfg.n_embd}
* **Attention Heads (Query / KV)**: {cfg.n_head} / {cfg.n_kv_head} (Grouped-Query Attention)
* **Context Length**: {cfg.block_size} tokens
* **Activation**: SwiGLU Gated Feed-Forward
* **Normalization**: Root Mean Square Normalization (RMSNorm)
* **Positional Encoding**: Rotary Position Embedding (RoPE)
"""
    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(model_card)
    print(f"  ✓ Exported: {readme_path}")
    print(f"\n🎉 Successfully exported Hugging Face artifacts to {output_dir}!\n")


def main():
    parser = argparse.ArgumentParser(description="AxiomLM Hugging Face Exporter CLI")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/model_latest.pt", help="Path to input PyTorch checkpoint .pt file")
    parser.add_argument("--output", type=str, default="exports/AxiomLM-124M-Systems", help="Output directory path for Hugging Face artifacts")
    parser.add_argument("--model_name", type=str, default="AxiomLM-124M-Systems", help="Model name for README model card")
    parser.add_argument("--license", type=str, default="mit", help="License identifier (e.g., mit, apache-2.0)")
    args = parser.parse_args()

    export_checkpoint_to_hf(
        checkpoint_path=args.checkpoint,
        output_dir=args.output,
        model_name=args.model_name,
        license_type=args.license,
    )


if __name__ == "__main__":
    main()
