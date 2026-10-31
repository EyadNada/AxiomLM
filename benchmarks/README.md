# AxiomLM Empirical Benchmark & Reproducibility Suite

> **Target Platform**: Apple Silicon M3 Pro SoC (12-Core CPU: 6P+6E, 18-Core Metal 4 GPU, 18 GB Unified LPDDR5 Memory @ 150 GB/s)
> **Software Stack**: macOS 15+ (Darwin 24+), Metal 4, PyTorch 2.x MPS, Apple MLX 0.32+, llama.cpp / ggml (0.27+)
> **Strict Baseline Parity**: Identical 124M LLaMA-3 architecture, $P=512$ prompt tokens, $N=128$ decode tokens, $B=1$ batch size, Metal GPU offload.

---

## 1. Overview & Academic Reproducibility Mission

To adhere to the rigorous experimental standards expected at top-tier systems venues (**MLSys, OSDI, EuroSys, ASPLOS**), all throughput and latency claims reported in AxiomLM are backed by an automated, one-click reproduction harness.

When comparing inference systems, reviewers rightfully scrutinize baseline parity:
* **Architecture Parity**: Comparing against identical model shapes ($L=12, D=768, H_Q=12, H_{KV}=4, D_{\text{ff}}=2048, V=50,304$).
* **Prompt & Decode Parity**: Standardizing prefill length ($P=512$) and generation length ($N=128$) across all frameworks rather than mixing short and long contexts.
* **Warmup & Synchronization**: Eliminating Metal pipeline compilation overhead via warmups, and enforcing explicit GPU stream synchronization (`torch.mps.synchronize()`, `mx.eval()`).
* **Hardware Telemetry Logging**: Auto-logging SoC characteristics, thermal state, commit hash, and software library versions in timestamped JSON manifests.

---

## 2. Controlled Benchmark Results (1:1 Parameter Parity)

*Measurements recorded on Apple M3 Pro (18 GPU Cores, 18 GB RAM @ 150 GB/s) with $P=512$ prompt tokens and $N=128$ decode tokens:*

| Framework / Engine | Precision | Decode Speed (tok/s) | Latency (ms/tok) | Speedup vs. MLX | Speedup vs. llama.cpp |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **PyTorch MPS Eager (Cached)** | FP16 | `370.3 ± 2.2` | `2.70 ms` | `1.24x` | `0.95x` |
| **Apple MLX Baseline** | FP16 | `298.6 ± 1.8` | `3.35 ms` | `1.00x` | `0.77x` |
| **llama.cpp Metal Baseline** | FP16 | `388.5 ± 1.0` | `2.57 ms` | `1.30x` | `1.00x` |
| **AxiomLM Modern Decode Runner** | **FP16** | **`413.4 ± 2.0`** | **`2.42 ms`** | **`1.38x`** | **`1.06x`** |
| **llama.cpp Metal Baseline** | Q4_0 | `690.2 ± 4.1` | `1.45 ms` | — | `1.00x` |
| **AxiomLM Quantized Metal GEMV** | **Q8_G64** | **`551.3 ± 4.6`** | **`1.81 ms`** | — | **`0.80x`** |
| **AxiomLM Quantized Metal GEMV** | **Q4_G64** | **`724.6 ± 1.5`** | **`1.38 ms`** | — | **`1.05x`** |
| **AxiomLM Speculative Batched QMM** | Q4_G64 (Python Loop, K=3) | `790.2 ± 12.3` | `1.27 ms` | — | `1.14x` |
| **AxiomLM Pure C++ Speculative Engine** | **Q4_G64 (C++ Loop, K=3)** | **`1,055.1 ± 0.8`** | **`0.95 ms`** | — | **`1.53x`** |

> *Note on Short-Context Peaks: In short contexts ($T < 192$) with minimal memory footprint, AxiomLM Q4_G64 reaches up to **`1,035.3 tok/s`**, Pure C++ Speculative Decoding reaches **`1,104.2 tok/s`**, and multi-token candidate verification compute rate reaches **`2,538.1 tok/s`** ($T=7$).*

---

## 3. Quickstart: Reproducing All Benchmarks

### 1-Click Master Reproduction Runner
To reproduce all numbers and output a fresh telemetry report, run:

```bash
./benchmarks/run_all_benchmarks.sh
```

Or invoke the Python orchestrator directly with custom parameters:

```bash
python3 benchmarks/reproduce_all.py --prompt 512 --gen 128 --repetitions 3
```

All results will be saved to `benchmarks/results/latest_benchmark_run.json`.

---

## 4. Standalone Framework Benchmark Harnesses

Every framework baseline has an isolated, self-contained harness with transparent CLI arguments:

### A. llama.cpp (`llama-bench`)
Automatically generates the architecture-parity GGUF models (`axiomlm_124m_f16.gguf` and `axiomlm_124m_q4.gguf`) and executes `llama-bench` on Metal:

```bash
python3 benchmarks/benchmark_llama_cpp.py --prompt 512 --gen 128 --repetitions 3
```

*Equivalent Raw CLI Invocation:*
```bash
llama-bench -m benchmarks/models/axiomlm_124m_q4.gguf -p 512 -n 128 -r 3 -ngl 99 -o json
```

### B. Apple MLX
Executes an exact 1:1 architecture implementation of AxiomLM in native Apple MLX using Metal GPU acceleration:

```bash
python3 benchmarks/benchmark_mlx.py --prompt 512 --gen 128 --repetitions 3
```

### C. AxiomLM
Executes AxiomLM across all optimization tiers (PyTorch MPS Cached, C++ Fused Decode Runner, Q8_G64, Q4_G64, and Speculative Decoding):

```bash
python3 benchmarks/benchmark_axiomlm.py --prompt 512 --gen 128 --repetitions 3
```

### D. Hardware & Git Telemetry
Dumps complete SoC architecture, CPU/GPU core counts, RAM bandwidth, and Git commit metadata:

```bash
python3 benchmarks/hardware_telemetry.py
```

---

## 5. Experimental Controls & Rigor

1. **JIT & Warmup Isolation**: All runs perform 5 warmup decoding cycles to force kernel compilation and Metal shader PSO caching before taking measurement timers.
2. **Explicit GPU Synchronization**:
   - PyTorch: `torch.mps.synchronize()` before and after measurement windows.
   - MLX: `mx.eval(tokens, caches)` to prevent lazy evaluation pipelining from skewing token latencies.
   - llama-bench: Native internal timing barriers.
3. **Statistical Validity**: All metrics report both arithmetic mean ($\mu$) and sample standard deviation ($\sigma$) across $R \ge 2$ repetitions.
4. **Data Integrity**: JSON reports record full system state, git commit hash, and configuration parameters to allow external researchers to verify claims under identical conditions.
