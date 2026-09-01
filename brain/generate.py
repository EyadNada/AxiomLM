"""
AxiomLM Standalone Interactive Inference & Generation CLI.

Supports:
- Checkpoint loading (.pt) and Pretrained Hugging Face GPT-2 weights
- Interactive chat/prompting shell
- Advanced sampling (Temperature, Top-k, Top-p Nucleus, Min-p, Repetition Penalty)
- Accelerated O(1) KV-Cache and naive O(T^2) modes
- Token-by-token live streaming and throughput telemetry (tok/s, ms/tok)
"""

import os
import sys
import time
import argparse
import torch
import torch.nn.functional as F
import tiktoken

# Ensure repo root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from brain.train_gpt2 import GPT, GPTConfig, sample_logits, get_raw_model


def load_model(
    checkpoint_path: str | None = None,
    pretrained: str | None = None,
    arch: str = "modern",
    device: str = "cpu",
) -> tuple[GPT, GPTConfig]:
    """Loads a model from a checkpoint or Hugging Face pretrained weights."""
    if pretrained:
        print(f"[AxiomLM] Loading pretrained weights from Hugging Face: {pretrained}")
        model = GPT.from_pretrained(pretrained)
        model.to(device)
        model.eval()
        return model, model.config

    if checkpoint_path:
        actual_path = checkpoint_path.strip()
        safetensors_file = None
        config_file = None

        if os.path.isdir(actual_path):
            st_candidate = os.path.join(actual_path, "model.safetensors")
            cfg_candidate = os.path.join(actual_path, "config.json")
            if os.path.isfile(st_candidate):
                safetensors_file = st_candidate
            if os.path.isfile(cfg_candidate):
                config_file = cfg_candidate
        elif os.path.isfile(actual_path) and actual_path.endswith(".safetensors"):
            safetensors_file = actual_path
            cfg_candidate = os.path.join(os.path.dirname(actual_path), "config.json")
            if os.path.isfile(cfg_candidate):
                config_file = cfg_candidate

        if safetensors_file:
            print(f"[AxiomLM] Loading safetensors model from: {safetensors_file}")
            from safetensors.torch import load_file
            import json
            weights = load_file(safetensors_file)

            if config_file and os.path.isfile(config_file):
                with open(config_file, "r", encoding="utf-8") as f:
                    cfg_data = json.load(f)
                config = GPTConfig(
                    block_size=cfg_data.get("n_positions", 1024),
                    vocab_size=cfg_data.get("vocab_size", 50304),
                    n_layer=cfg_data.get("n_layer", 12),
                    n_head=cfg_data.get("n_head", 12),
                    n_embd=cfg_data.get("n_embd", 768),
                    n_kv_head=cfg_data.get("n_kv_head", 4 if arch == "modern" else None),
                    norm_type=cfg_data.get("norm_type", "rmsnorm" if arch == "modern" else "layernorm"),
                    pos_emb=cfg_data.get("pos_emb", "rope" if arch == "modern" else "learned"),
                    mlp_type=cfg_data.get("mlp_type", "swiglu" if arch == "modern" else "gelu"),
                    bias=cfg_data.get("bias", False if arch == "modern" else True),
                )
            else:
                config = GPTConfig(
                    n_kv_head=4 if arch == "modern" else None,
                    norm_type="rmsnorm" if arch == "modern" else "layernorm",
                    pos_emb="rope" if arch == "modern" else "learned",
                    mlp_type="swiglu" if arch == "modern" else "gelu",
                    bias=False if arch == "modern" else True,
                )
            model = GPT(config)
            model.load_state_dict(weights)
            model.to(device)
            model.eval()
            print(f"[AxiomLM] Successfully restored safetensors model ({sum(p.numel() for p in model.parameters()):,} parameters)")
            return model, config

        if os.path.isfile(actual_path):
            print(f"[AxiomLM] Loading model checkpoint from: {actual_path}")
            checkpoint = torch.load(actual_path, map_location=device, weights_only=False)
            if "config" in checkpoint:
                config = checkpoint["config"]
            else:
                config = GPTConfig(
                    n_kv_head=4 if arch == "modern" else None,
                    norm_type="rmsnorm" if arch == "modern" else "layernorm",
                    pos_emb="rope" if arch == "modern" else "learned",
                    mlp_type="swiglu" if arch == "modern" else "gelu",
                    bias=False if arch == "modern" else True,
                )
            model = GPT(config)
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            model.to(device)
            model.eval()
            step = checkpoint.get("step", "N/A")
            print(f"[AxiomLM] Successfully restored checkpoint (Trained step: {step})")
            return model, config

    # Fallback to randomly initialized model
    print(f"[AxiomLM] Initializing un-trained {arch} architecture for demonstration...")
    config = GPTConfig(
        n_kv_head=4 if arch == "modern" else None,
        norm_type="rmsnorm" if arch == "modern" else "layernorm",
        pos_emb="rope" if arch == "modern" else "learned",
        mlp_type="swiglu" if arch == "modern" else "gelu",
        bias=False if arch == "modern" else True,
    )
    model = GPT(config).to(device)
    model.eval()
    return model, config


def stream_generate(
    model: GPT,
    enc,
    device: str,
    prompt: str,
    max_tokens: int = 60,
    temperature: float = 0.8,
    top_k: int | None = 50,
    top_p: float | None = 0.9,
    min_p: float | None = 0.05,
    repetition_penalty: float = 1.1,
    use_kv_cache: bool = True,
):
    """Generates and streams tokens to stdout in real-time with latency metrics."""
    tokens = enc.encode(prompt)
    if not tokens:
        tokens = [enc.eot_token]
    x = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

    print(f"\nPrompt: {prompt}", flush=True)
    print("Output: ", end="", flush=True)

    generated_tokens = x.clone()
    t0 = time.perf_counter()
    tokens_generated = 0

    with torch.no_grad():
        if use_kv_cache:
            # Prefill Phase
            kv_caches = [None] * model.config.n_layer
            logits, _, kv_caches = model(x, kv_caches=kv_caches)
            next_token = sample_logits(
                logits[:, -1, :],
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                min_p=min_p,
                repetition_penalty=repetition_penalty,
                prev_tokens=x,
            )
            generated_tokens = torch.cat((generated_tokens, next_token), dim=1)
            token_str = enc.decode([next_token.item()])
            print(token_str, end="", flush=True)
            tokens_generated += 1

            # Decode Phase (O(1) per token)
            while tokens_generated < max_tokens:
                logits, _, kv_caches = model(next_token, kv_caches=kv_caches)
                next_token = sample_logits(
                    logits[:, -1, :],
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    min_p=min_p,
                    repetition_penalty=repetition_penalty,
                    prev_tokens=generated_tokens,
                )
                generated_tokens = torch.cat((generated_tokens, next_token), dim=1)
                token_str = enc.decode([next_token.item()])
                print(token_str, end="", flush=True)
                tokens_generated += 1
                if next_token.item() == enc.eot_token:
                    break
        else:
            # Naive Eager Phase (O(T^2))
            while tokens_generated < max_tokens:
                logits, _ = model(generated_tokens)
                next_token = sample_logits(
                    logits[:, -1, :],
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    min_p=min_p,
                    repetition_penalty=repetition_penalty,
                    prev_tokens=generated_tokens,
                )
                generated_tokens = torch.cat((generated_tokens, next_token), dim=1)
                token_str = enc.decode([next_token.item()])
                print(token_str, end="", flush=True)
                tokens_generated += 1
                if next_token.item() == enc.eot_token:
                    break

    if device == "mps" and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()
    elif device == "cuda" and hasattr(torch.cuda, "synchronize"):
        torch.cuda.synchronize()

    total_time = time.perf_counter() - t0
    tok_per_sec = tokens_generated / max(total_time, 1e-6)
    ms_per_tok = (total_time / max(tokens_generated, 1)) * 1000

    print("\n" + "-" * 60)
    print(f"Metrics: {tokens_generated} tokens generated in {total_time*1000:.1f}ms | {tok_per_sec:.2f} tokens/s | {ms_per_tok:.2f} ms/token | Engine: {'KV-Cache O(1)' if use_kv_cache else 'Naive O(T^2)'}")
    print("-" * 60 + "\n")


def interactive_loop(model: GPT, enc, device: str, args: argparse.Namespace):
    """Runs an interactive REPL prompt session."""
    print("=" * 60)
    print(" AxiomLM Interactive Generation Console")
    print(f" Device: {device} | Sampling: temp={args.temperature}, top_k={args.top_k}, top_p={args.top_p}, min_p={args.min_p}")
    print(" Type your prompt and press Enter. Type 'exit' or 'quit' to exit.")
    print("=" * 60)

    while True:
        try:
            prompt = input("\n>>> ").strip()
            if not prompt:
                continue
            if prompt.lower() in {"exit", "quit"}:
                print("Exiting console.")
                break
            stream_generate(
                model=model,
                enc=enc,
                device=device,
                prompt=prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                min_p=args.min_p,
                repetition_penalty=args.repetition_penalty,
                use_kv_cache=not args.no_cache,
            )
        except (KeyboardInterrupt, EOFError):
            print("\nSession ended.")
            break


def main():
    parser = argparse.ArgumentParser(description="AxiomLM Interactive Text Generation & Inference CLI")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/model_latest.pt", help="Path to checkpoint .pt file")
    parser.add_argument("--pretrained", type=str, default=None, choices=["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"], help="Load official pretrained weights from Hugging Face")
    parser.add_argument("--arch", type=str, default="modern", choices=["classic", "modern"], help="Architecture spec")
    parser.add_argument("--prompt", type=str, default=None, help="Input prompt text. If omitted, starts interactive console")
    parser.add_argument("--max_tokens", type=int, default=60, help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature (0.0 for greedy argmax)")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k filtering threshold (None or 0 to disable)")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p Nucleus filtering threshold (None or 1.0 to disable)")
    parser.add_argument("--min_p", type=float, default=0.05, help="Min-p filtering threshold (None or 0.0 to disable)")
    parser.add_argument("--repetition_penalty", type=float, default=1.1, help="Repetition penalty multiplier (1.0 to disable)")
    parser.add_argument("--no_cache", action="store_true", help="Disable KV-cache and use naive eager recomputation")
    parser.add_argument("--device", type=str, default=None, help="Compute device ('mps', 'cuda', or 'cpu')")
    args = parser.parse_args()

    # Device auto-detection
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    enc = tiktoken.get_encoding("gpt2")
    model, _ = load_model(
        checkpoint_path=args.checkpoint if not args.pretrained else None,
        pretrained=args.pretrained,
        arch=args.arch,
        device=device,
    )

    if args.prompt:
        stream_generate(
            model=model,
            enc=enc,
            device=device,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            min_p=args.min_p,
            repetition_penalty=args.repetition_penalty,
            use_kv_cache=not args.no_cache,
        )
    else:
        interactive_loop(model, enc, device, args)


if __name__ == "__main__":
    main()
