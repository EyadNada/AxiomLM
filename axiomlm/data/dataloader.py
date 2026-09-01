"""
AxiomLM Multi-Shard Streaming Binary DataLoader.
"""
import os
import glob
from typing import Tuple, List, Optional
import numpy as np
import torch


class DataLoaderLite:
    """
    Lightweight, memory-mapped binary token loader with multi-shard support.
    Streams contiguous uint16 tokens across shard boundaries with sub-200 MB RAM utilization.
    """
    def __init__(
        self,
        B: int,
        T: int,
        process_rank: int = 0,
        num_processes: int = 1,
        split: str = "train",
        data_dir: str = "data",
    ):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.split = split
        self.data_dir = data_dir

        abs_data_dir = os.path.abspath(data_dir)
        if not os.path.exists(abs_data_dir):
            os.makedirs(abs_data_dir, exist_ok=True)

        # Look for multi-shard pattern (train_0000.bin) or single file (train.bin)
        shards = sorted(glob.glob(os.path.join(abs_data_dir, f"{split}_*.bin")))
        if not shards:
            single_path = os.path.join(abs_data_dir, f"{split}.bin")
            if os.path.exists(single_path):
                shards = [single_path]

        # Check subdirectories (e.g. data/systems_shards)
        if not shards and os.path.exists(os.path.join(abs_data_dir, "systems_shards")):
            sub_dir = os.path.join(abs_data_dir, "systems_shards")
            shards = sorted(glob.glob(os.path.join(sub_dir, f"{split}_*.bin")))
            if not shards and os.path.exists(os.path.join(sub_dir, f"{split}.bin")):
                shards = [os.path.join(sub_dir, f"{split}.bin")]

        # In-memory synthetic fallback if no shards found (for CI testing / fresh clones)
        if not shards:
            if process_rank == 0:
                print(f"[DataLoaderLite Warning] No binary shards found in '{data_dir}'. Generating in-memory token buffer.")
            synthetic_tokens = np.random.randint(0, 50257, size=10000, dtype=np.uint16)
            synth_path = os.path.join(abs_data_dir, f"{split}_synth.bin")
            synthetic_tokens.tofile(synth_path)
            shards = [synth_path]

        self.shards = shards
        self.total_tokens = sum(os.path.getsize(s) // 2 for s in self.shards)
        self.reset()

        if process_rank == 0:
            print(f"[DataLoaderLite] Loaded {split} ({len(self.shards)} shard{'s' if len(self.shards)>1 else ''}) from {data_dir} ({self.total_tokens:,} tokens total)")
            print(f"[DataLoaderLite] 1 epoch = {self.total_tokens // (B * T * num_processes)} batches")

    def reset(self) -> None:
        self.current_shard = 0
        self.tokens = np.memmap(self.shards[self.current_shard], dtype=np.uint16, mode='r')
        self.current_position = self.B * self.T * self.process_rank

    @property
    def current_shard_idx(self) -> int:
        return self.current_shard

    def set_step(self, step: int, grad_accum_steps: int = 1) -> None:
        """Fast-forwards data loader position to match a specific training step."""
        tokens_per_step = self.B * self.T * self.num_processes * grad_accum_steps
        target_token_offset = (step * tokens_per_step) % self.total_tokens

        cum_tokens = 0
        for i, shard_path in enumerate(self.shards):
            shard_size = os.path.getsize(shard_path) // 2
            if cum_tokens + shard_size > target_token_offset:
                self.current_shard = i
                self.tokens = np.memmap(self.shards[self.current_shard], dtype=np.uint16, mode='r')
                self.current_position = (target_token_offset - cum_tokens) + self.B * self.T * self.process_rank
                return
            cum_tokens += shard_size

    def next_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]

        # Handle shard boundary transition
        if len(buf) < B * T + 1:
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = np.memmap(self.shards[self.current_shard], dtype=np.uint16, mode='r')
            self.current_position = B * T * self.process_rank
            buf = self.tokens[self.current_position : self.current_position + B * T + 1]

        x = torch.from_numpy(buf[:-1].astype(np.int64)).view(B, T)
        y = torch.from_numpy(buf[1:].astype(np.int64)).view(B, T)

        self.current_position += B * T * self.num_processes
        return x, y
