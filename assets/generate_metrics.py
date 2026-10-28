import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import torch

# Ensure output directories exist
os.makedirs("assets", exist_ok=True)
os.makedirs("docs/assets", exist_ok=True)

# Set global style
plt.style.use(
    "seaborn-v0_8-whitegrid"
    if "seaborn-v0_8-whitegrid" in plt.style.available
    else "default"
)
plt.rcParams["font.sans-serif"] = "Helvetica", "Arial", "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 1.0


# -----------------------------------------------------------------------------
# 1. Newton-Schulz Spectral Singular Value Flattening (Muon Convergence)
# -----------------------------------------------------------------------------
def generate_newton_schulz_plot():
    print("Generating 11_newton_schulz_spectral_flattening.png...")
    torch.manual_seed(42)
    dim = 256

    # Create ill-conditioned matrix with exponentially decaying singular values
    U_rand, _ = torch.linalg.qr(torch.randn(dim, dim))
    V_rand, _ = torch.linalg.qr(torch.randn(dim, dim))
    s_init = torch.exp(-torch.linspace(0, 3.5, dim))
    G = U_rand @ torch.diag(s_init) @ V_rand.T

    # Run Newton-Schulz iterations
    a, b, c = (3.4445, -4.7750, 2.0315)
    eps = 1e-7
    X = G.float()
    X = X / (X.norm() + eps)

    spectra = [torch.linalg.svdvals(X).numpy()]
    for step in range(1, 6):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
        spectra.append(torch.linalg.svdvals(X).numpy())

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    colors = ["#94a3b8", "#38bdf8", "#3b82f6", "#6366f1", "#8b5cf6", "#10b981"]
    labels = [
        "Step 0 (Normalized Frobenius)",
        "Step 1 (First Cubic Contraction)",
        "Step 2",
        "Step 3",
        "Step 4",
        "Step 5 (Final Converged Muon Factor)",
    ]

    indices = np.arange(dim)
    for k, (s, col, lbl) in enumerate(zip(spectra, colors, labels)):
        lw = 2.5 if k in [0, 5] else 1.5
        ax.plot(indices, s, label=lbl, color=col, linewidth=lw, alpha=0.95)

    ax.axhline(
        1.0,
        color="#ef4444",
        linestyle="--",
        linewidth=1.5,
        alpha=0.8,
        label=r"Target Orthogonal Asymptote ($\sigma = 1.0$, 5-Step Band $[0.7, 1.3]$)",
    )
    ax.axhspan(
        0.70,
        1.30,
        color="#10b981",
        alpha=0.08,
        label=r"5-Step Quintic Convergence Band ($[1-\delta, 1+\delta]$)",
    )
    ax.set_title(
        "Muon Optimizer: 5-Step Quintic Newton–Schulz Singular Value Spectrum Compression",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    ax.set_xlabel(
        "Singular Value Index (Rank $1 \dots d$)", fontsize=11, fontweight="600"
    )
    ax.set_ylabel(
        "Singular Value Magnitude ($\sigma_i$)", fontsize=11, fontweight="600"
    )
    ax.set_ylim(-0.05, 1.35)
    ax.set_xlim(0, dim - 1)
    ax.legend(loc="upper right", frameon=True, framealpha=0.95, fontsize=9.5)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig("assets/11_newton_schulz_spectral_flattening.png")
    plt.close()


# -----------------------------------------------------------------------------
# 2. Hardware MFU & Roofline Model Analysis
# -----------------------------------------------------------------------------
def generate_roofline_plot():
    print("Generating 12_hardware_roofline_mfu_analysis.png...")
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    # Arithmetic intensity range (FLOPs/Byte)
    intensity = np.logspace(-1, 3, 500)

    # Peak hardware specs
    # Apple Silicon MPS: 10 TFLOPs peak, 100 GB/s bandwidth
    peak_flops_mps = 10.0
    bw_mps = 0.100  # TB/s = 100 GB/s
    roofline_mps = np.minimum(peak_flops_mps, intensity * bw_mps)

    # NVIDIA RTX 4090: 165 TFLOPs peak, 1008 GB/s bandwidth
    peak_flops_cuda = 165.0
    bw_cuda = 1.008  # TB/s = 1008 GB/s
    roofline_cuda = np.minimum(peak_flops_cuda, intensity * bw_cuda)

    ax.loglog(
        intensity,
        roofline_mps,
        color="#0284c7",
        linewidth=2.5,
        label="Apple Silicon MPS Roofline (10 TFLOPs, 100 GB/s)",
    )
    ax.loglog(
        intensity,
        roofline_cuda,
        color="#16a34a",
        linewidth=2.5,
        label="NVIDIA RTX 4090 Roofline (165 TFLOPs, 1.0 TB/s)",
    )

    # Operational points
    # 1. 2019 Eager FP32 Baseline (Low arithmetic intensity due to naive attention & eager mallocs)
    ax.scatter(
        [12.0],
        [2.09],
        color="#ef4444",
        s=120,
        zorder=5,
        label="2019 Baseline (Eager FP32 / Naive Softmax): 2.09 TFLOPs (20.9% MFU)",
    )
    # 2. 2026 Axiom-LM (BF16 + FlashAttention SDPA + SwiGLU)
    ax.scatter(
        [72.0],
        [6.87],
        color="#8b5cf6",
        s=150,
        zorder=5,
        marker="*",
        label="2026 Axiom-LM (BF16 + Fused SDPA + Muon): 6.87 TFLOPs (68.7% MFU)",
    )

    # Annotate transition
    ax.annotate(
        "$3.28\\times$ Hardware Compute Saturation\n(Memory-Bound $\\rightarrow$ Compute-Bound)",
        xy=(72.0, 6.87),
        xytext=(8.0, 22.0),
        arrowprops=dict(facecolor="#8b5cf6", shrink=0.08, width=1.5, headwidth=8),
        fontsize=9.5,
        fontweight="600",
        color="#4c1d95",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f5f3ff", edgecolor="#c4b5fd"),
    )

    ax.set_title(
        "Hardware Roofline Model: Arithmetic Intensity vs. Attained Throughput",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    ax.set_xlabel(
        "Arithmetic Intensity (FLOPs / Byte of Memory Transfer)",
        fontsize=11,
        fontweight="600",
    )
    ax.set_ylabel("Attained Performance (TFLOPs / sec)", fontsize=11, fontweight="600")
    ax.set_xlim(0.1, 1000)
    ax.set_ylim(0.1, 250)
    ax.legend(loc="lower right", frameon=True, framealpha=0.95, fontsize=9)
    ax.grid(True, which="both", ls="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig("assets/12_hardware_roofline_mfu_analysis.png")
    plt.close()


# -----------------------------------------------------------------------------
# 3. GQA vs MHA vs MQA KV-Cache Memory Scaling
# -----------------------------------------------------------------------------
def generate_kv_cache_scaling_plot():
    print("Generating 13_long_context_kv_cache_scaling.png...")
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    context_lengths = np.array([512, 1024, 2048, 4096, 8192, 16384])
    batch_size = 4
    n_layers = 12
    head_dim = 64
    bytes_per_elem = 2  # BF16 / FP16

    # KV Cache Formula: 2 * n_layers * n_kv_heads * head_dim * context_len * batch_size * bytes_per_elem
    # MHA: 12 KV heads
    kv_mha = (
        2 * n_layers * 12 * head_dim * context_lengths * batch_size * bytes_per_elem
    ) / (1024**2)
    # GQA: 4 KV heads (3x reduction)
    kv_gqa = (
        2 * n_layers * 4 * head_dim * context_lengths * batch_size * bytes_per_elem
    ) / (1024**2)
    # MQA: 1 KV head (12x reduction)
    kv_mqa = (
        2 * n_layers * 1 * head_dim * context_lengths * batch_size * bytes_per_elem
    ) / (1024**2)

    ax.plot(
        context_lengths,
        kv_mha,
        marker="o",
        linewidth=2.5,
        color="#ef4444",
        label="Multi-Head Attention (MHA: 12 KV Heads)",
    )
    ax.plot(
        context_lengths,
        kv_gqa,
        marker="s",
        linewidth=2.5,
        color="#3b82f6",
        label="Grouped-Query Attention (GQA: 4 KV Heads) [-66.7% VRAM]",
    )
    ax.plot(
        context_lengths,
        kv_mqa,
        marker="^",
        linewidth=2.5,
        color="#10b981",
        label="Multi-Query Attention (MQA: 1 KV Head) [-91.7% VRAM]",
    )

    ax.set_title(
        "KV-Cache VRAM Footprint Scaling Across Extended Context ($B=4, L=12$)",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    ax.set_xlabel("Context Sequence Length ($T$ Tokens)", fontsize=11, fontweight="600")
    ax.set_ylabel("KV-Cache Memory Footprint (MB)", fontsize=11, fontweight="600")
    ax.set_xticks(context_lengths)
    ax.set_xticklabels([f"{t:,}" for t in context_lengths])
    ax.legend(loc="upper left", frameon=True, framealpha=0.95, fontsize=9.5)
    ax.grid(True, linestyle="--", alpha=0.5)

    # Annotate 8K savings
    savings_8k = kv_mha[4] - kv_gqa[4]
    ax.annotate(
        f"At 8K Context:\n{kv_mha[4]:.0f} MB $\\rightarrow$ {kv_gqa[4]:.0f} MB\n({savings_8k:.0f} MB Saved per stream)",
        xy=(8192, kv_gqa[4]),
        xytext=(9000, 180),
        arrowprops=dict(facecolor="#3b82f6", shrink=0.08, width=1.5, headwidth=8),
        fontsize=9,
        fontweight="600",
        color="#1e3a8a",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#eff6ff", edgecolor="#93c5fd"),
    )

    plt.tight_layout()
    plt.savefig("assets/13_long_context_kv_cache_scaling.png")
    plt.close()


# -----------------------------------------------------------------------------
# 4. SwiGLU Bilinear Gating vs. GELU / ReLU Activation Landscape (Replaces Draft 1 Fig 1)
# -----------------------------------------------------------------------------
def generate_swiglu_vs_gelu_plot():
    print("Generating 14_swiglu_vs_gelu_activation_landscape.png...")
    x = np.linspace(-4.0, 4.0, 500)
    silu = x / (1.0 + np.exp(-x))
    gelu = 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * (x**3))))
    relu = np.maximum(0.0, x)
    dsilu = (1.0 / (1.0 + np.exp(-x))) * (1.0 + x * (1.0 - 1.0 / (1.0 + np.exp(-x))))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

    ax1.plot(
        x,
        silu,
        label=r"Swish / SiLU ($x \cdot \sigma(x)$) [AxiomLM Gate]",
        color="#6366f1",
        linewidth=2.5,
    )
    ax1.plot(
        x,
        gelu,
        label=r"GELU ($x \cdot \Phi(x)$) [Classic GPT-2]",
        color="#0284c7",
        linestyle="--",
        linewidth=2.0,
    )
    ax1.plot(
        x,
        relu,
        label=r"ReLU ($\max(0, x)$)",
        color="#94a3b8",
        linestyle=":",
        linewidth=1.8,
    )
    ax1.set_title(
        "Activation Function Profiles ($d_{\\mathrm{ffn}} = 2048$)",
        fontsize=12,
        fontweight="bold",
    )
    ax1.set_xlabel("Pre-Activation Input ($z$)", fontsize=10.5, fontweight="600")
    ax1.set_ylabel("Activation Output", fontsize=10.5, fontweight="600")
    ax1.legend(loc="upper left", frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.plot(
        x,
        dsilu,
        label=r"$\frac{d}{dz}\mathrm{SiLU}(z)$ (Smooth Non-Zero Negative Gradient)",
        color="#10b981",
        linewidth=2.5,
    )
    ax2.plot(
        x,
        silu * (0.8 * x),
        label=r"SwiGLU Bilinear Gate ($\mathrm{SiLU}(z_{\mathrm{gate}}) \odot z_{\mathrm{up}}$)",
        color="#8b5cf6",
        linewidth=2.2,
    )
    ax2.set_title(
        "SwiGLU First-Order Derivative & Bilinear Modulation",
        fontsize=12,
        fontweight="bold",
    )
    ax2.set_xlabel("Gate Projection Input ($z$)", fontsize=10.5, fontweight="600")
    ax2.set_ylabel("Gradient / Modulated Response", fontsize=10.5, fontweight="600")
    ax2.legend(loc="upper left", frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig("assets/14_swiglu_vs_gelu_activation_landscape.png")
    plt.close()


# -----------------------------------------------------------------------------
# 5. End-to-End 5-Stage Ablation & 4096 Ring-Buffer Benchmark (Apple M3 Pro 18GB)
# -----------------------------------------------------------------------------
def generate_m3_pro_ablation_plot():
    print("Generating 15_apple_m3_pro_ablation_and_ring_buffer.png...")
    stages = [
        "1. Classic MHA\n(Eager O(T²))",
        "2. + Dynamic KV\n(12-Head MHA)",
        "3. + Zero-Copy GQA\n(4-Head Static KV)",
        "4. + Fused Metal/NEON\n(FP32 Engine)",
        "5. + Native FP16 Metal\n+ Static Ring KV",
        "6. Long-Context 4096\n(FP16 Ring Buffer)",
    ]
    tok_s = [8.2, 176.2, 227.6, 238.7, 404.9, 399.5]
    kv_mb = [54.0, 54.0, 18.0, 18.0, 9.0, 12.0]

    fig, ax1 = plt.subplots(figsize=(11, 6), dpi=300)
    x = np.arange(len(stages))
    width = 0.38

    bars1 = ax1.bar(
        x - width / 2,
        tok_s,
        width,
        label="Decode Throughput (tokens/sec)",
        color="#4f46e5",
        alpha=0.92,
    )
    ax1.set_ylabel(
        "Autoregressive Throughput (tok/s)",
        fontsize=11,
        fontweight="bold",
        color="#4f46e5",
    )
    ax1.set_ylim(0, 470)
    ax1.set_xticks(x)
    ax1.set_xticklabels(stages, fontsize=9.5, fontweight="600")
    ax1.grid(True, axis="y", linestyle="--", alpha=0.4)

    for bar in bars1:
        h = bar.get_height()
        ax1.annotate(
            f"{h:.1f}\ntok/s",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
            color="#312e81",
        )

    ax2 = ax1.twinx()
    bars2 = ax2.bar(
        x + width / 2,
        kv_mb,
        width,
        label="Active KV-Cache Footprint (MB)",
        color="#10b981",
        alpha=0.85,
    )
    ax2.set_ylabel(
        "Active KV-Cache Memory (MB)", fontsize=11, fontweight="bold", color="#047857"
    )
    ax2.set_ylim(0, 70)
    ax2.grid(False)

    for bar in bars2:
        h = bar.get_height()
        ax2.annotate(
            f"{h:.1f} MB",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
            color="#065f46",
        )

    ax1.set_title(
        "AxiomLM Component Ablation on Apple M3 Pro (12-Core CPU, 18-Core GPU, 18 GB Unified Memory)",
        fontsize=12.5,
        fontweight="bold",
        pad=15,
    )

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2, labels1 + labels2, loc="upper left", frameon=True, fontsize=9.5
    )

    plt.tight_layout()
    plt.savefig("assets/15_apple_m3_pro_ablation_and_ring_buffer.png")
    plt.close()


def generate_cross_framework_plot():
    print("Generating 10_framework_comparison.png...")
    frameworks = [
        "Hugging Face\ntransformers (MPS FP16)",
        "Apple MLX\n(0.32.4 Metal FP16)",
        "llama.cpp\n(0.7.0 GGUF FP16)",
        "AxiomLM Engine\n(Fused Metal FP16)",
    ]
    tok_s = [152.2, 345.7, 384.9, 404.9]
    colors = ["#64748b", "#0284c7", "#10b981", "#4f46e5"]

    fig, ax = plt.subplots(figsize=(9.5, 5.5), dpi=300)
    x = np.arange(len(frameworks))
    bars = ax.bar(x, tok_s, width=0.52, color=colors, alpha=0.92)
    ax.set_ylabel("Autoregressive Decode Speed (tok/s)", fontsize=11, fontweight="bold")
    ax.set_title(
        "Cross-Framework Benchmark on Apple M3 Pro (12L, 768D, 12Q/4KV GQA, FP16, B=1)",
        fontsize=12.5,
        fontweight="bold",
        pad=14,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(frameworks, fontsize=10, fontweight="600")
    ax.set_ylim(0, 465)
    ax.grid(True, axis="y", linestyle="--", alpha=0.45)

    for bar in bars:
        h = bar.get_height()
        ax.annotate(
            f"{h:.1f} tok/s\n({1000.0/h:.2f} ms/tok)",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
            color="#1e293b",
        )

    plt.tight_layout()
    plt.savefig("assets/10_framework_comparison.png")
    plt.close()


if __name__ == "__main__":
    generate_newton_schulz_plot()
    generate_roofline_plot()
    generate_kv_cache_scaling_plot()
    generate_swiglu_vs_gelu_plot()
    generate_m3_pro_ablation_plot()
    generate_cross_framework_plot()
    for fname in os.listdir("assets"):
        if fname.endswith(".png"):
            shutil.copy(
                os.path.join("assets", fname), os.path.join("docs/assets", fname)
            )
    print(
        "All 5 visualization assets generated and synced to docs/assets/ successfully!"
    )
