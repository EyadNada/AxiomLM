import torch
import axiomlm as ax

def main():
    print("=== AxiomLM Standalone Modules Example ===")
    batch_size, seq_len, dim = 2, 64, 512
    x = torch.randn(batch_size, seq_len, dim)

    # 1. Apply Root Mean Square Normalization (RMSNorm)
    rmsnorm = ax.RMSNorm(dim=dim)
    x_norm = rmsnorm(x)
    print(f"1. RMSNorm Output Shape: {x_norm.shape}")

    # 2. Apply SwiGLU Feed-Forward Network
    swiglu = ax.SwiGLUMLP(dim=dim, hidden_dim=int(dim * (8/3)))
    x_ffn = swiglu(x_norm)
    print(f"2. SwiGLU Output Shape:  {x_ffn.shape}")

    # 3. Apply Rotary Position Embeddings (RoPE)
    freqs_cis = ax.precompute_rope_frequencies(dim=64, end=1024)
    q = torch.randn(batch_size, seq_len, 8, 64) # 8 Query heads
    k = torch.randn(batch_size, seq_len, 2, 64) # 2 KV heads (GQA)
    
    q_rope, k_rope = ax.apply_rope(q, k, freqs_cis)
    print(f"3. RoPE Query Shape:     {q_rope.shape}")
    print(f"   RoPE Key Shape:       {k_rope.shape}")

if __name__ == "__main__":
    main()
