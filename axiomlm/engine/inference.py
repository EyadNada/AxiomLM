"""
AxiomLM Accelerated Inference Engine & Text Generation CLI.
"""
from typing import Optional, List, Tuple, Generator, Union, Any
import os
import sys
import time
import argparse
import json
import torch
import torch.nn.functional as F
import tiktoken
from .paged_cache import PagedKVCache, SequenceContext


from ..models.transformer import Transformer, ModelConfig, GPT, GPTConfig


def sample_logits(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = 50,
    top_p: Optional[float] = None,
    min_p: Optional[float] = None,
    repetition_penalty: float = 1.0,
    prev_tokens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Applies temperature scaling, repetition penalty, Top-K, Min-P, and Top-P (Nucleus) truncation.
    Returns: Next token index tensor of shape (B, 1).
    """
    logits = logits.clone()

    if repetition_penalty != 1.0 and prev_tokens is not None:
        B, S = prev_tokens.shape
        for b in range(B):
            unique_toks = torch.unique(prev_tokens[b])
            tok_logits = logits[b, unique_toks]
            penalized = torch.where(tok_logits > 0, tok_logits / repetition_penalty, tok_logits * repetition_penalty)
            logits[b, unique_toks] = penalized

    if temperature <= 1e-4:
        return torch.argmax(logits, dim=-1, keepdim=True)

    logits = logits / temperature

    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        val, _ = torch.topk(logits, k)
        logits[logits < val[:, [-1]]] = -float('Inf')

    probs = F.softmax(logits, dim=-1)

    if min_p is not None and min_p > 0.0:
        p_max = probs.max(dim=-1, keepdim=True).values
        cutoff = p_max * min_p
        probs = torch.where(probs < cutoff, torch.zeros_like(probs), probs)
        probs_sum = probs.sum(dim=-1, keepdim=True)
        probs = torch.where(probs_sum > 0, probs / probs_sum, F.softmax(logits, dim=-1))

    if top_p is not None and top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        sorted_probs[sorted_indices_to_remove] = 0.0
        probs = torch.scatter(torch.zeros_like(probs), -1, sorted_indices, sorted_probs)
        probs_sum = probs.sum(dim=-1, keepdim=True)
        probs = torch.where(probs_sum > 0, probs / probs_sum, F.softmax(logits, dim=-1))

    next_tok = torch.multinomial(probs, 1)
    return next_tok


def generate_samples(
    model: Transformer,
    enc: Any,
    device: str,
    prompt: str = "import torch\n",
    num_samples: int = 2,
    max_length: int = 40,
    temperature: float = 1.0,
    top_k: Optional[int] = 50,
    top_p: Optional[float] = None,
    min_p: Optional[float] = None,
    repetition_penalty: float = 1.0,
) -> List[str]:
    """Generates autoregressive text samples using standard eager re-computation (O(T^2))."""
    model.eval()
    tokens = enc.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0).repeat(num_samples, 1)

    while tokens.size(1) < max_length:
        with torch.no_grad():
            logits, _ = model(tokens)
            next_token = sample_logits(
                logits[:, -1, :],
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                min_p=min_p,
                repetition_penalty=repetition_penalty,
                prev_tokens=tokens,
            )
            tokens = torch.cat((tokens, next_token), dim=1)

    samples = []
    for i in range(num_samples):
        sample_text = enc.decode(tokens[i, :max_length].tolist())
        samples.append(sample_text)
    return samples


def generate_with_cache(
    model: Transformer,
    enc: Any,
    device: str,
    prompt: str = "import torch\n",
    num_samples: int = 1,
    max_length: int = 40,
    temperature: float = 1.0,
    top_k: Optional[int] = 50,
    top_p: Optional[float] = None,
    min_p: Optional[float] = None,
    repetition_penalty: float = 1.0,
) -> List[str]:
    """
    Accelerated autoregressive text generation using per-layer Key-Value caching.
    Reduces compute complexity from O(T^2) to O(1) per decoding step.
    """
    model.eval()
    prompt_tokens = enc.encode(prompt)
    x = torch.tensor(prompt_tokens, dtype=torch.long, device=device).unsqueeze(0).repeat(num_samples, 1)

    with torch.no_grad():
        if num_samples == 1:
            paged_cache = PagedKVCache(
                num_blocks=1024,
                block_size=16,
                n_layer=model.config.n_layer,
                n_kv_head=model.config.n_kv_head or model.config.n_head,
                head_dim=model.config.n_embd // model.config.n_head,
                dtype=torch.float32,
                device=device
            )
            seq_context = SequenceContext(paged_cache)
            kv_caches = [seq_context] * model.config.n_layer
        else:
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
        generated_tokens = torch.cat((x, next_token), dim=1)

        while generated_tokens.size(1) < max_length:
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

    samples = []
    for i in range(num_samples):
        sample_text = enc.decode(generated_tokens[i, :max_length].tolist())
        samples.append(sample_text)
    return samples


def benchmark_generation_speed(
    model: Transformer,
    enc: Any,
    device: str,
    prompt: str = "import torch\n",
    max_length: int = 100,
) -> None:
    """Benchmarks generation throughput (tokens/sec) comparing Naive O(T^2) vs KV-Cache O(1)."""
    model.eval()
    print(f"\n[Axiom-LM Benchmark] Benchmarking generation to {max_length} tokens on {device}...")

    # Warmup
    _ = generate_samples(model, enc, device, prompt=prompt, num_samples=1, max_length=20)
    _ = generate_with_cache(model, enc, device, prompt=prompt, num_samples=1, max_length=20)

    # 1. Benchmark Naive Eager O(T^2)
    if device == "mps" and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()
    elif device == "cuda" and hasattr(torch.cuda, "synchronize"):
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    _ = generate_samples(model, enc, device, prompt=prompt, num_samples=1, max_length=max_length)
    if device == "mps" and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()
    elif device == "cuda" and hasattr(torch.cuda, "synchronize"):
        torch.cuda.synchronize()
    t_naive = time.perf_counter() - t0

    # 2. Benchmark KV-Cache O(1)
    t0 = time.perf_counter()
    _ = generate_with_cache(model, enc, device, prompt=prompt, num_samples=1, max_length=max_length)
    if device == "mps" and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()
    elif device == "cuda" and hasattr(torch.cuda, "synchronize"):
        torch.cuda.synchronize()
    t_cache = time.perf_counter() - t0

    speedup = t_naive / t_cache if t_cache > 0 else 1.0
    print(f"  • Naive Eager O(T^2) Time:  {t_naive:.4f}s ({max_length / t_naive:.2f} tok/s)")
    print(f"  • KV-Cache Engine O(1) Time: {t_cache:.4f}s ({max_length / t_cache:.2f} tok/s)")
    print(f"  • KV-Cache Acceleration:    {speedup:.2f}x Faster\n")


def load_model(
    checkpoint_path: Optional[str] = None,
    pretrained: Optional[str] = None,
    arch: str = "modern",
    device: str = "cpu",
) -> Tuple[Transformer, ModelConfig]:
    """Loads a model from a checkpoint, directory, or Hugging Face pretrained weights."""
    if pretrained:
        print(f"[AxiomLM] Loading pretrained weights from Hugging Face: {pretrained}")
        model = Transformer.from_pretrained(pretrained)
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
            
            if config_file and os.path.exists(config_file):
                with open(config_file, "r") as f:
                    cfg_dict = json.load(f)
                config = ModelConfig(
                    block_size=cfg_dict.get("max_position_embeddings", 1024),
                    vocab_size=cfg_dict.get("vocab_size", 50304),
                    n_layer=cfg_dict.get("num_hidden_layers", 12),
                    n_head=cfg_dict.get("num_attention_heads", 12),
                    n_embd=cfg_dict.get("hidden_size", 768),
                    n_kv_head=cfg_dict.get("num_key_value_heads", 4 if arch == "modern" else None),
                    norm_type="rmsnorm" if arch == "modern" else "layernorm",
                    pos_emb="rope" if arch == "modern" else "learned",
                    mlp_type="swiglu" if arch == "modern" else "gelu",
                    bias=False if arch == "modern" else True,
                )
            else:
                config = ModelConfig(
                    n_kv_head=4 if arch == "modern" else None,
                    norm_type="rmsnorm" if arch == "modern" else "layernorm",
                    pos_emb="rope" if arch == "modern" else "learned",
                    mlp_type="swiglu" if arch == "modern" else "gelu",
                    bias=False if arch == "modern" else True,
                )

            model = Transformer(config)
            state_dict = load_file(safetensors_file, device=device)
            model.load_state_dict(state_dict, strict=False)
            model.to(device)
            model.eval()
            return model, config

        if os.path.isfile(actual_path) and actual_path.endswith(".pt"):
            print(f"[AxiomLM] Loading PyTorch checkpoint from: {actual_path}")
            ckpt = torch.load(actual_path, map_location=device, weights_only=False)
            if "config" in ckpt:
                config = ckpt["config"]
            else:
                config = ModelConfig(
                    n_kv_head=4 if arch == "modern" else None,
                    norm_type="rmsnorm" if arch == "modern" else "layernorm",
                    pos_emb="rope" if arch == "modern" else "learned",
                    mlp_type="swiglu" if arch == "modern" else "gelu",
                    bias=False if arch == "modern" else True,
                )
            model = Transformer(config)
            raw_sd = ckpt["model"] if "model" in ckpt else ckpt
            cleaned_sd = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in raw_sd.items()}
            model.load_state_dict(cleaned_sd, strict=False)
            model.to(device)
            model.eval()
            return model, config

    raise FileNotFoundError(f"Could not resolve model checkpoint or safetensors file at: {checkpoint_path}")


class InferenceEngine:
    """
    High-level, user-friendly Python Inference Engine for AxiomLM models.
    """
    def __init__(
        self,
        model: Union[Transformer, str],
        device: Optional[str] = None,
        tokenizer: Optional[Any] = None,
    ):
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device

        if isinstance(model, str):
            self.model, self.config = load_model(checkpoint_path=model, device=device)
        else:
            self.model = model.to(device)
            self.config = model.config

        if tokenizer is None:
            self.tokenizer = tiktoken.get_encoding("gpt2")
        else:
            self.tokenizer = tokenizer

    def generate(
        self,
        prompt: str,
        max_tokens: int = 50,
        temperature: float = 0.8,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.9,
        min_p: Optional[float] = 0.05,
        repetition_penalty: float = 1.1,
    ) -> str:
        """Generates text from a prompt using O(1) KV-caching."""
        results = generate_with_cache(
            model=self.model,
            enc=self.tokenizer,
            device=self.device,
            prompt=prompt,
            num_samples=1,
            max_length=len(self.tokenizer.encode(prompt)) + max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
        )
        return results[0]

    def stream(
        self,
        prompt: str,
        max_tokens: int = 50,
        temperature: float = 0.8,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.9,
        min_p: Optional[float] = 0.05,
        repetition_penalty: float = 1.1,
    ) -> Generator[str, None, None]:
        """Yields generated tokens one by one as they are decoded."""
        self.model.eval()
        prompt_tokens = self.tokenizer.encode(prompt)
        x = torch.tensor(prompt_tokens, dtype=torch.long, device=self.device).unsqueeze(0)

        with torch.no_grad():
            paged_cache = PagedKVCache(
                num_blocks=1024,
                block_size=16,
                n_layer=self.config.n_layer,
                n_kv_head=self.config.n_kv_head or self.config.n_head,
                head_dim=self.config.n_embd // self.config.n_head,
                dtype=torch.float32,
                device=self.device
            )
            seq_context = SequenceContext(paged_cache)
            kv_caches = [seq_context] * self.config.n_layer
            
            logits, _, kv_caches = self.model(x, kv_caches=kv_caches)

            next_token = sample_logits(
                logits[:, -1, :],
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                min_p=min_p,
                repetition_penalty=repetition_penalty,
                prev_tokens=x,
            )
            yield self.tokenizer.decode([next_token.item()])

            tokens_so_far = torch.cat((x, next_token), dim=1)
            for _ in range(max_tokens - 1):
                logits, _, kv_caches = self.model(next_token, kv_caches=kv_caches)
                next_token = sample_logits(
                    logits[:, -1, :],
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    min_p=min_p,
                    repetition_penalty=repetition_penalty,
                    prev_tokens=tokens_so_far,
                )
                yield self.tokenizer.decode([next_token.item()])
                tokens_so_far = torch.cat((tokens_so_far, next_token), dim=1)


def main():
    parser = argparse.ArgumentParser(description="AxiomLM Text Generation CLI")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/model_latest.pt")
    parser.add_argument("--pretrained", type=str, default=None)
    parser.add_argument("--arch", type=str, default="modern")
    parser.add_argument("--prompt", type=str, default="import torch\nimport torch.nn as nn\n")
    parser.add_argument("--max_tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--min_p", type=float, default=0.05)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    engine = InferenceEngine(model=args.checkpoint if not args.pretrained else args.pretrained, device=args.device)
    print(f"\n[AxiomLM Prompt]: {args.prompt}\n[AxiomLM Generation]:\n")
    for token in engine.stream(
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        min_p=args.min_p,
        repetition_penalty=args.repetition_penalty,
    ):
        sys.stdout.write(token)
        sys.stdout.flush()
    print("\n")


if __name__ == "__main__":
    main()
