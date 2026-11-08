"""
Native HuggingFace Safetensors Open-Weight Importer for AxiomLM.

Supports LLaMA-3 / LLaMA-3.2 (`meta-llama/Llama-3.2-1B`, `unsloth/Llama-3.2-1B`),
SmolLM2 (`HuggingFaceTB/SmolLM2-360M`), and Qwen2.5 (`Qwen/Qwen2.5-0.5B`) architectures.
Performs exact RoPE complex-pair weight/bias permutation and fused QKV packing
directly from `.safetensors` shards without intermediate checkpoint conversion.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from ..models.transformer import ModelConfig, Transformer


def permute_hf_rope_weight(
    w: torch.Tensor, n_heads: int, head_dim: int
) -> torch.Tensor:
    """
    Permutes HuggingFace half-split RoPE projection weights `[0:hd//2, hd//2:hd]`
    into AxiomLM's interleaved complex-pair RoPE layout `[0, hd//2, 1, hd//2+1, ...]`.
    """
    return (
        w.view(n_heads, 2, head_dim // 2, -1)
        .transpose(1, 2)
        .reshape(n_heads * head_dim, -1)
        .contiguous()
    )


def unpermute_hf_rope_weight(
    w: torch.Tensor, n_heads: int, head_dim: int
) -> torch.Tensor:
    """
    Inverse of `permute_hf_rope_weight`: converts interleaved complex-pair RoPE
    weights back into HuggingFace half-split `[0:hd//2, hd//2:hd]` layout.
    """
    return (
        w.view(n_heads, head_dim // 2, 2, -1)
        .transpose(1, 2)
        .reshape(n_heads * head_dim, -1)
        .contiguous()
    )


def permute_hf_rope_bias(b: torch.Tensor, n_heads: int, head_dim: int) -> torch.Tensor:
    """
    Permutes HuggingFace half-split RoPE projection biases `[0:hd//2, hd//2:hd]`
    (e.g. Qwen2.5 Q/K biases) into AxiomLM's interleaved complex-pair layout.
    """
    return (
        b.view(n_heads, 2, head_dim // 2)
        .transpose(1, 2)
        .reshape(n_heads * head_dim)
        .contiguous()
    )


def resolve_hf_model_dir(repo_id_or_path: str | Path, token: str | None = None) -> Path:
    """
    Resolves a local model directory or downloads `.safetensors` + `config.json`
    from HuggingFace Hub. Honors `HF_TOKEN` for gated repositories.
    """
    candidate = Path(repo_id_or_path).expanduser()
    if candidate.is_dir():
        return candidate

    from huggingface_hub import snapshot_download

    auth_token = token if token is not None else os.environ.get("HF_TOKEN")
    try:
        local_dir = snapshot_download(
            repo_id=str(repo_id_or_path),
            allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"],
            token=auth_token,
        )
        return Path(local_dir)
    except Exception as exc:
        msg = str(exc)
        if "401" in msg or "403" in msg or "GatedRepo" in type(exc).__name__:
            raise RuntimeError(
                f"Access to gated HuggingFace repository '{repo_id_or_path}' requires authentication. "
                f"Export HF_TOKEN=<your_token> or use an ungated open-weight repository such as "
                f"'unsloth/Llama-3.2-1B', 'HuggingFaceTB/SmolLM2-360M', or 'Qwen/Qwen2.5-0.5B'."
            ) from exc
        raise


def build_model_config_from_hf(
    hf_cfg: dict[str, Any],
    has_qkv_bias: bool = False,
    use_fused_kernels: bool = True,
    max_seq_len: int | None = None,
) -> ModelConfig:
    """
    Translates a HuggingFace `config.json` dictionary (LlamaConfig / Qwen2Config)
    into an exact `AxiomLM.ModelConfig`.
    """
    n_head = int(hf_cfg["num_attention_heads"])
    n_kv_head = int(hf_cfg.get("num_key_value_heads", n_head))
    n_embd = int(hf_cfg["hidden_size"])
    n_layer = int(hf_cfg["num_hidden_layers"])
    vocab_size = int(hf_cfg["vocab_size"])
    intermediate_size = int(hf_cfg["intermediate_size"])
    rms_norm_eps = float(hf_cfg.get("rms_norm_eps", 1e-6))
    tie_word_embeddings = bool(hf_cfg.get("tie_word_embeddings", True))

    rope_params = hf_cfg.get("rope_parameters") or {}
    rope_theta = float(hf_cfg.get("rope_theta", rope_params.get("rope_theta", 10000.0)))
    rope_scaling = hf_cfg.get("rope_scaling")
    if (
        rope_scaling is None
        and isinstance(rope_params, dict)
        and (rope_params.get("rope_type") not in (None, "default"))
    ):
        rope_scaling = rope_params

    cfg_max_pos = int(hf_cfg.get("max_position_embeddings", 4096))
    block_size = max_seq_len if max_seq_len is not None else min(cfg_max_pos, 4096)

    qkv_bias = bool(
        has_qkv_bias
        or hf_cfg.get("attention_bias", False)
        or hf_cfg.get("qkv_bias", False)
    )

    return ModelConfig(
        block_size=block_size,
        vocab_size=vocab_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        n_kv_head=n_kv_head,
        norm_type="rmsnorm",
        pos_emb="rope",
        mlp_type="swiglu",
        bias=False,
        rope_theta=rope_theta,
        use_fused_kernels=use_fused_kernels,
        intermediate_size=intermediate_size,
        rms_norm_eps=rms_norm_eps,
        tie_word_embeddings=tie_word_embeddings,
        qkv_bias=qkv_bias,
        rope_scaling=rope_scaling,
    )


class _SafetensorsReader:
    """Unified lazy tensor reader over single or sharded `.safetensors` files."""

    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        index_path = model_dir / "model.safetensors.index.json"
        single_path = model_dir / "model.safetensors"
        self._handles: dict[str, Any] = {}
        self.weight_map: dict[str, str] = {}

        if index_path.exists():
            with open(index_path, "r", encoding="utf-8") as f:
                idx_data = json.load(f)
            self.weight_map = dict(idx_data["weight_map"])
        elif single_path.exists():
            handle = safe_open(str(single_path), framework="pt", device="cpu")
            self._handles["model.safetensors"] = handle
            for k in handle.keys():
                self.weight_map[k] = "model.safetensors"
        else:
            st_files = sorted(model_dir.glob("*.safetensors"))
            if not st_files:
                raise FileNotFoundError(f"No .safetensors files found in {model_dir}")
            for st_file in st_files:
                handle = safe_open(str(st_file), framework="pt", device="cpu")
                self._handles[st_file.name] = handle
                for k in handle.keys():
                    self.weight_map[k] = st_file.name

    def __contains__(self, key: str) -> bool:
        return key in self.weight_map

    def get_tensor(self, key: str, dtype: torch.dtype) -> torch.Tensor:
        if key not in self.weight_map:
            raise KeyError(f"Tensor '{key}' not found in safetensors archive")
        shard_name = self.weight_map[key]
        if shard_name not in self._handles:
            shard_path = self.model_dir / shard_name
            self._handles[shard_name] = safe_open(
                str(shard_path), framework="pt", device="cpu"
            )
        return self._handles[shard_name].get_tensor(key).to(dtype=dtype)


def load_hf_safetensors(
    repo_id_or_path: str | Path,
    device: str | torch.device = "mps",
    dtype: torch.dtype = torch.float16,
    use_fused_kernels: bool = True,
    max_seq_len: int | None = None,
    token: str | None = None,
) -> Transformer:
    """
    Loads a HuggingFace Safetensors model (`LlamaForCausalLM` / `Qwen2ForCausalLM`)
    directly into an `AxiomLM.Transformer` with zero random weight initialization overhead.
    """
    model_dir = resolve_hf_model_dir(repo_id_or_path, token=token)
    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json in {model_dir}")

    with open(config_path, "r", encoding="utf-8") as f:
        hf_cfg = json.load(f)

    reader = _SafetensorsReader(model_dir)
    has_qkv_bias = "model.layers.0.self_attn.q_proj.bias" in reader
    if "lm_head.weight" in reader and not bool(hf_cfg.get("tie_word_embeddings", True)):
        hf_cfg["tie_word_embeddings"] = False

    config = build_model_config_from_hf(
        hf_cfg,
        has_qkv_bias=has_qkv_bias,
        use_fused_kernels=use_fused_kernels,
        max_seq_len=max_seq_len,
    )

    model = Transformer(config, init_weights=False)
    model = model.to(dtype=dtype)

    n_head = config.n_head
    n_kv_head = config.n_kv_head if config.n_kv_head is not None else n_head
    head_dim = config.n_embd // n_head

    with torch.no_grad():
        wte = reader.get_tensor("model.embed_tokens.weight", dtype=dtype)
        model.transformer.wte.weight.copy_(wte)
        if not config.tie_word_embeddings and "lm_head.weight" in reader:
            lmh = reader.get_tensor("lm_head.weight", dtype=dtype)
            model.lm_head.weight.copy_(lmh)

        model.transformer.ln_f.weight.copy_(
            reader.get_tensor("model.norm.weight", dtype=dtype)
        )

        for i, block in enumerate(model.transformer.h):
            prefix = f"model.layers.{i}"
            block.ln_1.weight.copy_(
                reader.get_tensor(f"{prefix}.input_layernorm.weight", dtype=dtype)
            )
            block.ln_2.weight.copy_(
                reader.get_tensor(
                    f"{prefix}.post_attention_layernorm.weight", dtype=dtype
                )
            )

            wq = permute_hf_rope_weight(
                reader.get_tensor(f"{prefix}.self_attn.q_proj.weight", dtype=dtype),
                n_head,
                head_dim,
            )
            wk = permute_hf_rope_weight(
                reader.get_tensor(f"{prefix}.self_attn.k_proj.weight", dtype=dtype),
                n_kv_head,
                head_dim,
            )
            wv = reader.get_tensor(f"{prefix}.self_attn.v_proj.weight", dtype=dtype)
            block.attn.c_attn.weight.copy_(torch.cat([wq, wk, wv], dim=0))

            if block.attn.c_attn.bias is not None and (
                f"{prefix}.self_attn.q_proj.bias" in reader
            ):
                bq = permute_hf_rope_bias(
                    reader.get_tensor(f"{prefix}.self_attn.q_proj.bias", dtype=dtype),
                    n_head,
                    head_dim,
                )
                bk = permute_hf_rope_bias(
                    reader.get_tensor(f"{prefix}.self_attn.k_proj.bias", dtype=dtype),
                    n_kv_head,
                    head_dim,
                )
                bv = reader.get_tensor(f"{prefix}.self_attn.v_proj.bias", dtype=dtype)
                block.attn.c_attn.bias.copy_(torch.cat([bq, bk, bv], dim=0))

            block.attn.c_proj.weight.copy_(
                reader.get_tensor(f"{prefix}.self_attn.o_proj.weight", dtype=dtype)
            )
            block.mlp.w_gate.weight.copy_(
                reader.get_tensor(f"{prefix}.mlp.gate_proj.weight", dtype=dtype)
            )
            block.mlp.w_up.weight.copy_(
                reader.get_tensor(f"{prefix}.mlp.up_proj.weight", dtype=dtype)
            )
            block.mlp.w_down.weight.copy_(
                reader.get_tensor(f"{prefix}.mlp.down_proj.weight", dtype=dtype)
            )

    target_device = torch.device(device)
    model = model.to(device=target_device, dtype=dtype)
    model.eval()
    return model
