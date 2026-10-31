"""
AxiomLM GGUF Export & Quantization Utility for Standardized llama-bench Parity.
Exports an exact architecture-parity LLaMA-3/AxiomLM 124M model to GGUF format
to enable direct, apples-to-apples baseline comparisons via llama.cpp (llama-bench).
"""

import argparse
import os
import shutil
import subprocess
import sys

import numpy as np

try:
    import gguf
except ImportError:
    print("[Error] gguf package not installed. Install via pip install gguf")
    sys.exit(1)


def export_axiomlm_gguf(
    output_f16_path: str,
    n_layer: int = 12,
    n_embd: int = 768,
    n_head: int = 12,
    n_kv_head: int = 4,
    ffn_dim: int = 2048,
    vocab_size: int = 50304,
    context_length: int = 1024,
) -> None:
    """Exports an unquantized FP16 GGUF model with exact AxiomLM architecture hyperparameters."""
    os.makedirs(os.path.dirname(os.path.abspath(output_f16_path)), exist_ok=True)
    print(f"[GGUF Export] Writing FP16 model to: {output_f16_path}")

    writer = gguf.GGUFWriter(output_f16_path, "llama")
    writer.add_name("AxiomLM-124M")
    writer.add_context_length(context_length)
    writer.add_embedding_length(n_embd)
    writer.add_block_count(n_layer)
    writer.add_feed_forward_length(ffn_dim)
    writer.add_head_count(n_head)
    writer.add_head_count_kv(n_kv_head)
    writer.add_layer_norm_rms_eps(1e-5)
    writer.add_rope_dimension_count(64)
    writer.add_rope_freq_base(10000.0)

    # Tokenizer tokens (standard BPE / GPT-2 / LLaMA vocab indices)
    tokens = [f"<tok_{i}>" for i in range(vocab_size)]
    scores = [0.0] * vocab_size
    tok_types = [1] * vocab_size
    writer.add_tokenizer_model("llama")
    writer.add_token_list(tokens)
    writer.add_token_scores(scores)
    writer.add_token_types(tok_types)
    writer.add_bos_token_id(1)
    writer.add_eos_token_id(2)

    # Weights initialization (deterministic zeros/ones for microbenchmarking)
    head_dim = n_embd // n_head
    kv_dim = n_kv_head * head_dim

    writer.add_tensor(
        "token_embd.weight", np.zeros((vocab_size, n_embd), dtype=np.float16)
    )
    writer.add_tensor("output_norm.weight", np.ones((n_embd,), dtype=np.float32))
    writer.add_tensor("output.weight", np.zeros((vocab_size, n_embd), dtype=np.float16))

    for i in range(n_layer):
        writer.add_tensor(
            f"blk.{i}.attn_norm.weight", np.ones((n_embd,), dtype=np.float32)
        )
        writer.add_tensor(
            f"blk.{i}.attn_q.weight", np.zeros((n_embd, n_embd), dtype=np.float16)
        )
        writer.add_tensor(
            f"blk.{i}.attn_k.weight", np.zeros((kv_dim, n_embd), dtype=np.float16)
        )
        writer.add_tensor(
            f"blk.{i}.attn_v.weight", np.zeros((kv_dim, n_embd), dtype=np.float16)
        )
        writer.add_tensor(
            f"blk.{i}.attn_output.weight", np.zeros((n_embd, n_embd), dtype=np.float16)
        )
        writer.add_tensor(
            f"blk.{i}.ffn_norm.weight", np.ones((n_embd,), dtype=np.float32)
        )
        writer.add_tensor(
            f"blk.{i}.ffn_gate.weight", np.zeros((ffn_dim, n_embd), dtype=np.float16)
        )
        writer.add_tensor(
            f"blk.{i}.ffn_up.weight", np.zeros((ffn_dim, n_embd), dtype=np.float16)
        )
        writer.add_tensor(
            f"blk.{i}.ffn_down.weight", np.zeros((n_embd, ffn_dim), dtype=np.float16)
        )

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(
        f"[GGUF Export] Successfully exported FP16 model ({os.path.getsize(output_f16_path) / (1024*1024):.2f} MB)"
    )


def quantize_gguf_model(
    input_f16_path: str, output_q4_path: str, quant_type: str = "Q4_0"
) -> None:
    """Invokes llama-quantize to produce a Q4_0 quantized GGUF artifact."""
    quantize_bin = shutil.which("llama-quantize")
    if not quantize_bin:
        raise RuntimeError(
            "llama-quantize executable not found in PATH. Install via brew install llama.cpp"
        )

    print(
        f"[GGUF Quantize] Quantizing {input_f16_path} -> {output_q4_path} ({quant_type})..."
    )
    cmd = [quantize_bin, input_f16_path, output_q4_path, quant_type]
    subprocess.run(cmd, check=True)
    print(
        f"[GGUF Quantize] Successfully generated {quant_type} model ({os.path.getsize(output_q4_path) / (1024*1024):.2f} MB)"
    )


def prepare_benchmark_models(force: bool = False) -> dict:
    """Prepares both FP16 and Q4_0 GGUF benchmark models, caching if present."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, "models")
    f16_path = os.path.join(models_dir, "axiomlm_124m_f16.gguf")
    q4_path = os.path.join(models_dir, "axiomlm_124m_q4.gguf")

    if force or not os.path.exists(f16_path):
        export_axiomlm_gguf(f16_path)
    else:
        print(f"[GGUF Cache] Using existing FP16 model: {f16_path}")

    if force or not os.path.exists(q4_path):
        quantize_gguf_model(f16_path, q4_path, "Q4_0")
    else:
        print(f"[GGUF Cache] Using existing Q4_0 model: {q4_path}")

    return {
        "f16_path": f16_path,
        "q4_path": q4_path,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export GGUF models for llama-bench parity"
    )
    parser.add_argument(
        "--force", action="store_true", help="Force re-export even if models exist"
    )
    args = parser.parse_args()
    prepare_benchmark_models(force=args.force)
