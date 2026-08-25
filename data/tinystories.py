"""
Axiom-LM: High-Performance Data Tokenization & Sharding Pipeline
Downloads and tokenizes TinyStories using tiktoken (gpt2 BPE) and saves as
compact, zero-overhead memory-mappable uint16 binary files for Apple Silicon Unified Memory.
"""

import os
import sys
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

import argparse

def prepare_tinystories(target_tokens: int = 20_000_000, val_ratio: float = 0.05, force: bool = False):
    """
    Downloads and tokenizes the TinyStories dataset.
    Target: ~20M tokens (~19M train, ~1M val).
    Saves: data/train.bin and data/val.bin as uint16 raw arrays.
    """
    train_bin_path = os.path.join(DATA_DIR, "train.bin")
    val_bin_path = os.path.join(DATA_DIR, "val.bin")

    if not force and os.path.exists(train_bin_path) and os.path.exists(val_bin_path):
        print(f"[Axiom-LM] Binary shards already exist in {DATA_DIR}:")
        print(f"  - Train: {train_bin_path} ({os.path.getsize(train_bin_path):,} bytes)")
        print(f"  - Val:   {val_bin_path} ({os.path.getsize(val_bin_path):,} bytes)")
        return

    print("[Axiom-LM] Loading TinyStories dataset from Hugging Face...")
    dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)

    enc = tiktoken.get_encoding("gpt2")
    eot = enc.eot_token  # 50256 (<|endoftext|>)

    val_target = int(target_tokens * val_ratio)
    train_target = target_tokens - val_target

    print(f"[Axiom-LM] Tokenizing {target_tokens:,} tokens ({train_target:,} train, {val_target:,} val)...")
    
    all_tokens = []
    pbar = tqdm(total=target_tokens, unit="tokens", desc="Tokenizing")

    for item in dataset:
        text = item["text"]
        # Prefix each story with <|endoftext|> delimiter
        tokens = [eot] + enc.encode_ordinary(text)
        all_tokens.extend(tokens)
        pbar.update(len(tokens))
        if len(all_tokens) >= target_tokens:
            break
    pbar.close()

    all_tokens = all_tokens[:target_tokens]
    val_tokens = np.array(all_tokens[:val_target], dtype=np.uint16)
    train_tokens = np.array(all_tokens[val_target:], dtype=np.uint16)

    print(f"[Axiom-LM] Writing binary shards (uint16)...")
    train_tokens.tofile(train_bin_path)
    val_tokens.tofile(val_bin_path)

    print(f"[Axiom-LM] Successfully created:")
    print(f"  - Train: {train_bin_path} ({len(train_tokens):,} tokens, {os.path.getsize(train_bin_path) / 1024 / 1024:.2f} MB)")
    print(f"  - Val:   {val_bin_path} ({len(val_tokens):,} tokens, {os.path.getsize(val_bin_path) / 1024 / 1024:.2f} MB)")

    # Explicitly clean up streaming connections to avoid socket teardown warnings
    del dataset
    del all_tokens

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Axiom-LM TinyStories Dataset Tokenizer & Sharder")
    parser.add_argument("--target_tokens", type=int, default=20_000_000, help="Target total token count (default: 20M)")
    parser.add_argument("--val_ratio", type=float, default=0.05, help="Validation split ratio (default: 0.05)")
    parser.add_argument("--force", action="store_true", help="Force re-tokenization even if binary shards exist")
    args = parser.parse_args()

    prepare_tinystories(target_tokens=args.target_tokens, val_ratio=args.val_ratio, force=args.force)
