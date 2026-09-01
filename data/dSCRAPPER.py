"""
AxiomLM: High-Performance Multi-Shard Systems ML & GPU Kernel Dataset Builder.
"""
import os
import sys
import argparse
import numpy as np
import tiktoken
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

SYSTEMS_KERNEL_SNIPPETS = [
    '''
# OpenAI Triton Fused RMSNorm Forward and Backward Kernel
import torch
import triton
import triton.language as tl

@triton.jit
def _rmsnorm_fwd_kernel(X_ptr, Y_ptr, W_ptr, stride_row, N: tl.constexpr, eps: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    x = tl.load(X_ptr + row_idx * stride_row + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    variance = tl.sum(x * x, axis=0) / N
    rsqrt = 1.0 / tl.sqrt(variance + eps)
    y = x * rsqrt * w
    tl.store(Y_ptr + row_idx * stride_row + cols, y.to(tl.float16), mask=mask)

def triton_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    M, N = x.shape
    y = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(N)
    _rmsnorm_fwd_kernel[(M,)](x, y, weight, x.stride(0), N=N, eps=eps, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)
    return y
''',
    '''
# OpenAI Triton Fused SwiGLU Gated Activation Kernel
import torch
import triton
import triton.language as tl

@triton.jit
def _swiglu_fwd_kernel(Gate_ptr, Up_ptr, Out_ptr, stride_m, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    gate_vals = tl.load(Gate_ptr + row_idx * stride_m + cols, mask=mask, other=0.0).to(tl.float32)
    up_vals = tl.load(Up_ptr + row_idx * stride_m + cols, mask=mask, other=0.0).to(tl.float32)
    silu_gate = gate_vals * (1.0 / (1.0 + tl.exp(-gate_vals)))
    out = silu_gate * up_vals
    tl.store(Out_ptr + row_idx * stride_m + cols, out.to(tl.float16), mask=mask)

def triton_swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    M, N = gate.shape
    out = torch.empty_like(gate)
    BLOCK_SIZE = triton.next_power_of_2(N)
    _swiglu_fwd_kernel[(M,)](gate, up, out, gate.stride(0), N=N, BLOCK_SIZE=BLOCK_SIZE, num_warps=8)
    return out
''',
    '''
# OpenAI Triton FlashAttention Tiled Forward Operator with Online Softmax Rescaling
import torch
import triton
import triton.language as tl

@triton.jit
def _flash_attn_fwd_kernel(
    Q, K, V, Out, sm_scale,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, N_CTX, BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    q_ptrs = Q + off_hz * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    k_ptrs = K + off_hz * stride_kh + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
    v_ptrs = V + off_hz * stride_vh + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
    o_ptrs = Out + off_hz * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)
    for start_n in range(0, (start_m + 1) * BLOCK_M, BLOCK_N):
        k = tl.load(k_ptrs + start_n * stride_kn, mask=(start_n + offs_n[:, None]) < N_CTX, other=0.0)
        qk = tl.dot(q, tl.trans(k)) * sm_scale
        qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf"))
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        alpha = tl.exp(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        v = tl.load(v_ptrs + start_n * stride_vn, mask=(start_n + offs_n[:, None]) < N_CTX, other=0.0)
        acc += tl.dot(p.to(tl.float16), v)
        m_i = m_ij
    acc = acc / l_i[:, None]
    tl.store(o_ptrs, acc.to(tl.float16), mask=offs_m[:, None] < N_CTX)
''',
    '''
# Systems ML: Arithmetic Intensity and Roofline Analysis Model
class RooflineModel:
    def __init__(self, peak_tflops: float, memory_bandwidth_gb_s: float):
        self.peak_flops = peak_tflops * 1e12
        self.bandwidth = memory_bandwidth_gb_s * 1e9
        self.ridge_point = self.peak_flops / self.bandwidth

    def analyze_operator(self, name: str, flops: float, bytes_transferred: float):
        intensity = flops / bytes_transferred
        is_memory_bound = intensity < self.ridge_point
        return {
            "operator": name,
            "arithmetic_intensity": f"{intensity:.2f} FLOPs/Byte",
            "regime": "MEMORY_BOUND" if is_memory_bound else "COMPUTE_BOUND",
        }
''',
]


def build_systems_dataset(
    target_tokens: int = 50_000_000,
    shard_size_tokens: int = 10_000_000,
    val_ratio: float = 0.05,
    output_dir: str = "data/systems_shards",
    use_huggingface_stream: bool = True,
):
    os.makedirs(output_dir, exist_ok=True)
    enc = tiktoken.get_encoding("gpt2")
    eot = enc.eot_token

    val_total_target = int(target_tokens * val_ratio)
    train_total_target = target_tokens - val_total_target

    print(f"\n[AxiomLM Systems Dataset Builder] Target: {target_tokens:,} tokens")
    print(f"  -> Train: {train_total_target:,} | Val: {val_total_target:,}")
    print(f"  -> Shard size: {shard_size_tokens:,} tokens per shard")

    all_tokens = []

    # Tokenize and inject curated systems kernel snippets with priority
    curated_tokens = []
    for snippet in SYSTEMS_KERNEL_SNIPPETS:
        curated_tokens.extend([eot] + enc.encode_ordinary(snippet.strip()))

    repeats = max(1, min(100, target_tokens // (len(curated_tokens) * 4)))
    for _ in range(repeats):
        all_tokens.extend(curated_tokens)

    pbar = tqdm(total=target_tokens, initial=len(all_tokens), unit="tokens", desc="Tokenizing")

    if use_huggingface_stream:
        try:
            from datasets import load_dataset
            print("Streaming from Hugging Face (HuggingFaceTB/smollm-corpus -> python-edu)...")
            ds = load_dataset("HuggingFaceTB/smollm-corpus", "python-edu", split="train", streaming=True)
            for sample in ds:
                text = sample.get("text", "")
                if text and len(text.strip()) > 40:
                    toks = [eot] + enc.encode_ordinary(text)
                    all_tokens.extend(toks)
                    pbar.update(len(toks))
                    if len(all_tokens) >= target_tokens:
                        break
        except Exception as e:
            print(f"[Warning] Streaming error: {e}. Using fallback...")

    # If stream ended or offline, use curated snippets as the pure code fallback
    if len(all_tokens) < target_tokens:
        print("[Warning] Could not reach target tokens from HuggingFace. Duplicating curated systems kernels to fill the remaining quota to ensure a pure code dataset...")
        while len(all_tokens) < target_tokens:
            all_tokens.extend(curated_tokens)
            pbar.update(min(len(curated_tokens), target_tokens - len(all_tokens)))

    pbar.close()
    all_tokens = all_tokens[:target_tokens]

    val_tokens = all_tokens[:val_total_target]
    train_tokens = all_tokens[val_total_target:]

    val_np = np.array(val_tokens, dtype=np.uint16)
    val_np.tofile(os.path.join(output_dir, "val_0000.bin"))
    val_np.tofile(os.path.join(output_dir, "val.bin"))

    num_train_shards = max(1, int(np.ceil(len(train_tokens) / shard_size_tokens)))
    for shard_idx in range(num_train_shards):
        start_idx = shard_idx * shard_size_tokens
        end_idx = min(len(train_tokens), (shard_idx + 1) * shard_size_tokens)
        shard_data = np.array(train_tokens[start_idx:end_idx], dtype=np.uint16)
        shard_path = os.path.join(output_dir, f"train_{shard_idx:04d}.bin")
        shard_data.tofile(shard_path)
        print(f"  ✓ Train Shard {shard_idx+1}/{num_train_shards}: {shard_path} ({len(shard_data):,} tokens)")

    if num_train_shards == 1:
        np.array(train_tokens, dtype=np.uint16).tofile(os.path.join(output_dir, "train.bin"))

    print(f"Successfully generated {num_train_shards} training shards in {output_dir}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AxiomLM Multi-Shard Dataset Builder")
    parser.add_argument("--target_tokens", type=int, default=10_000_000)
    parser.add_argument("--shard_size", type=int, default=5_000_000)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--output_dir", type=str, default="data/systems_shards")
    parser.add_argument("--no_hf", action="store_true")
    args = parser.parse_args()

    build_systems_dataset(
        target_tokens=args.target_tokens,
        shard_size_tokens=args.shard_size,
        val_ratio=args.val_ratio,
        output_dir=args.output_dir,
        use_huggingface_stream=not args.no_hf,
    )
