"""
AxiomLM Hugging Face Safetensors Model Exporter.
"""

import os
import json
import argparse
from typing import Dict
import torch
import re

from ..models.transformer import ModelConfig


def export_checkpoint_to_hf(
    checkpoint_path: str,
    output_dir: str,
    model_name: str = "AxiomLM-124M-Modern",
    license_type: str = "mit",
) -> None:
    """
    Loads an AxiomLM PyTorch .pt checkpoint, converts its tensors to standard Hugging Face
    Llama (Modern) or GPT2 (Classic) formats, and exports all required assets.
    """
    try:
        from safetensors.torch import save_file
    except ImportError:
        raise ImportError(
            "Package 'safetensors' is required. Install via `pip install safetensors`."
        )

    try:
        from transformers import GPT2TokenizerFast
    except ImportError:
        GPT2TokenizerFast = None  # type: ignore

    os.makedirs(output_dir, exist_ok=True)
    print("=" * 70)
    print(" AxiomLM: Hugging Face .safetensors Model Exporter")
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
        print(
            "    Config not found in checkpoint dict. Using default modern specification."
        )
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

    if "model_state_dict" in checkpoint:
        raw_state_dict = checkpoint["model_state_dict"]
    elif "model" in checkpoint:
        raw_state_dict = checkpoint["model"]
    else:
        raw_state_dict = checkpoint

    cleaned_state_dict: Dict[str, torch.Tensor] = {}
    is_modern = cfg.norm_type == "rmsnorm"

    for k, v in raw_state_dict.items():
        if not isinstance(v, torch.Tensor):
            continue
        clean_key = k.replace("_orig_mod.", "").replace("module.", "")
        if clean_key.endswith(".attn.bias") or clean_key.endswith(".freqs_cis"):
            continue

        # ---------------------------------------------------------
        # NATIVE HUGGING FACE ARCHITECTURE TRANSLATION
        # ---------------------------------------------------------
        if is_modern:
            # Map AxiomLM Modern -> Hugging Face LlamaForCausalLM
            if clean_key == "transformer.wte.weight":
                cleaned_state_dict["model.embed_tokens.weight"] = v
            elif clean_key == "transformer.ln_f.weight":
                cleaned_state_dict["model.norm.weight"] = v
            elif clean_key == "lm_head.weight":
                cleaned_state_dict["lm_head.weight"] = v
            else:
                match = re.match(r"transformer\.h\.(\d+)\.(.*)", clean_key)
                if match:
                    layer_idx = match.group(1)
                    sub_key = match.group(2)

                    if sub_key == "ln_1.weight":
                        cleaned_state_dict[
                            f"model.layers.{layer_idx}.input_layernorm.weight"
                        ] = v
                    elif sub_key == "ln_2.weight":
                        cleaned_state_dict[
                            f"model.layers.{layer_idx}.post_attention_layernorm.weight"
                        ] = v
                    elif sub_key == "mlp.w_gate.weight":
                        cleaned_state_dict[
                            f"model.layers.{layer_idx}.mlp.gate_proj.weight"
                        ] = v
                    elif sub_key == "mlp.w_up.weight":
                        cleaned_state_dict[
                            f"model.layers.{layer_idx}.mlp.up_proj.weight"
                        ] = v
                    elif sub_key == "mlp.w_down.weight":
                        cleaned_state_dict[
                            f"model.layers.{layer_idx}.mlp.down_proj.weight"
                        ] = v
                    elif sub_key == "attn.c_proj.weight":
                        cleaned_state_dict[
                            f"model.layers.{layer_idx}.self_attn.o_proj.weight"
                        ] = v
                    elif sub_key == "attn.c_attn.weight":
                        # Split c_attn into q_proj, k_proj, v_proj
                        head_dim = cfg.n_embd // cfg.n_head
                        n_kv_head = (
                            cfg.n_kv_head if cfg.n_kv_head is not None else cfg.n_head
                        )
                        q_dim = cfg.n_head * head_dim
                        kv_dim = n_kv_head * head_dim

                        cleaned_state_dict[
                            f"model.layers.{layer_idx}.self_attn.q_proj.weight"
                        ] = v[:q_dim, :]
                        cleaned_state_dict[
                            f"model.layers.{layer_idx}.self_attn.k_proj.weight"
                        ] = v[q_dim : q_dim + kv_dim, :]
                        cleaned_state_dict[
                            f"model.layers.{layer_idx}.self_attn.v_proj.weight"
                        ] = v[q_dim + kv_dim :, :]
        else:
            # Map AxiomLM Classic -> Hugging Face GPT2LMHeadModel
            # GPT-2 in HF expects linear weights in Conv1D format [in_features, out_features]
            if (
                "attn.c_attn.weight" in clean_key
                or "attn.c_proj.weight" in clean_key
                or "mlp.c_fc.weight" in clean_key
                or "mlp.c_proj.weight" in clean_key
            ):
                cleaned_state_dict[clean_key] = v.t()
            else:
                cleaned_state_dict[clean_key] = v

    # 1. Export model.safetensors
    safetensors_path = os.path.join(output_dir, "model.safetensors")
    save_file(cleaned_state_dict, safetensors_path)
    total_params = sum(p.numel() for p in cleaned_state_dict.values())
    filesize_mb = os.path.getsize(safetensors_path) / (1024 * 1024)
    print(
        f"\n  ✓ Exported: {safetensors_path} ({total_params:,} parameters, {filesize_mb:.2f} MB)"
    )

    # 2. Export config.json
    n_kv_head = (
        cfg.n_kv_head if getattr(cfg, "n_kv_head", None) is not None else cfg.n_head
    )
    if is_modern:
        hf_config = {
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "vocab_size": cfg.vocab_size,
            "hidden_size": cfg.n_embd,
            "num_hidden_layers": cfg.n_layer,
            "num_attention_heads": cfg.n_head,
            "num_key_value_heads": n_kv_head,
            "intermediate_size": int(2 * (4 * cfg.n_embd) / 3),
            "max_position_embeddings": cfg.block_size,
            "rms_norm_eps": getattr(cfg, "eps", 1e-6),
            "rope_theta": getattr(cfg, "rope_theta", 10000.0),
            "torch_dtype": "float32",
        }
    else:
        hf_config = {
            "architectures": ["GPT2LMHeadModel"],
            "model_type": "gpt2",
            "vocab_size": cfg.vocab_size,
            "n_embd": cfg.n_embd,
            "n_layer": cfg.n_layer,
            "n_head": cfg.n_head,
            "n_positions": cfg.block_size,
            "n_inner": 4 * cfg.n_embd,
            "torch_dtype": "float32",
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
            print(
                "  ✓ Exported: Complete Hugging Face tokenizer assets (vocab.json, merges.txt, tokenizer.json)"
            )
        except Exception as e:
            print(f"    Notice: Tokenizer files could not be saved automatically: {e}")

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

AxiomLM 124M autoregressive language model trained using the **Muon (5-step Newton-Schulz) optimizer** and bare-metal fused kernels.
Fully native Hugging Face compatibility.

## Model Specifications
* **Parameters**: ~{total_params / 1e6:.1f}M
* **Layers**: {cfg.n_layer}
* **Hidden Size**: {cfg.n_embd}
* **Attention Heads (Query / KV)**: {cfg.n_head} / {n_kv_head}
* **Context Length**: {cfg.block_size} tokens
"""
    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(model_card)
    print(f"  ✓ Exported: {readme_path}")
    print(f"\n Successfully exported Hugging Face artifacts to {output_dir}!\n")


def main():
    parser = argparse.ArgumentParser(description="AxiomLM Hugging Face Exporter CLI")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/model_latest.pt",
        help="Path to input PyTorch checkpoint .pt file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="exports/AxiomLM-124M-Systems",
        help="Output directory path for Hugging Face artifacts",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="AxiomLM-124M-Systems",
        help="Model name for README model card",
    )
    parser.add_argument(
        "--license",
        type=str,
        default="mit",
        help="License identifier (e.g., mit, apache-2.0)",
    )
    args = parser.parse_args()

    export_checkpoint_to_hf(
        checkpoint_path=args.checkpoint,
        output_dir=args.output,
        model_name=args.model_name,
        license_type=args.license,
    )


if __name__ == "__main__":
    main()
