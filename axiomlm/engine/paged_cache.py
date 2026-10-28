import torch
from typing import List, Tuple


class PagedKVCache:
    """
    vLLM-style block table memory manager for KV-Cache.
    Eliminates memory fragmentation by storing KV states in non-contiguous memory blocks.
    """

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        n_layer: int,
        n_kv_head: int,
        head_dim: int,
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
    ):
        self.block_size = block_size
        self.num_blocks = num_blocks

        # Shape: [n_layer, 2 (K, V), num_blocks, n_kv_head, block_size, head_dim]
        self.cache = torch.zeros(
            (n_layer, 2, num_blocks, n_kv_head, block_size, head_dim),
            dtype=dtype,
            device=device,
        )
        self.free_blocks = list(range(num_blocks - 1, -1, -1))

    def allocate_block(self) -> int:
        if not self.free_blocks:
            raise RuntimeError("PagedKVCache Out of Memory: No free blocks available.")
        return self.free_blocks.pop()


class SequenceContext:
    def __init__(self, cache: PagedKVCache):
        self.cache = cache
        self.block_table: List[int] = []
        self.layer_seq_lens = {}

    @property
    def seq_len(self):
        return max(self.layer_seq_lens.values()) if self.layer_seq_lens else 0

    def get_reconstructed_cache(
        self, layer_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reconstructs the contiguous (K, V) tensors for F.scaled_dot_product_attention.
        Returns K, V of shape [1, seq_len, n_kv_head, head_dim]
        """
        current_len = self.layer_seq_lens.get(layer_idx, 0)
        if current_len == 0:
            return None, None

        k_blocks = []
        v_blocks = []

        for b_idx in self.block_table:
            k_blocks.append(self.cache.cache[layer_idx, 0, b_idx])
            v_blocks.append(self.cache.cache[layer_idx, 1, b_idx])

        k_full = torch.cat(
            k_blocks, dim=1
        )  # [n_kv_head, total_allocated_len, head_dim]
        v_full = torch.cat(v_blocks, dim=1)

        k_out = k_full[:, :current_len, :].transpose(0, 1).unsqueeze(0)
        v_out = v_full[:, :current_len, :].transpose(0, 1).unsqueeze(0)

        return k_out, v_out

    def append_kv(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor):
        num_new_tokens = k.size(1)
        k = k.squeeze(0).transpose(0, 1)
        v = v.squeeze(0).transpose(0, 1)

        current_len = self.layer_seq_lens.get(layer_idx, 0)
        new_len = current_len + num_new_tokens

        # Allocate blocks based on the first layer processed
        if layer_idx == 0 or not self.block_table:
            required_blocks = (
                new_len + self.cache.block_size - 1
            ) // self.cache.block_size
            while len(self.block_table) < required_blocks:
                self.block_table.append(self.cache.allocate_block())

        tokens_written = 0
        while tokens_written < num_new_tokens:
            block_idx = current_len // self.cache.block_size
            offset = current_len % self.cache.block_size

            space_in_block = self.cache.block_size - offset
            tokens_to_write = min(space_in_block, num_new_tokens - tokens_written)

            physical_block = self.block_table[block_idx]

            self.cache.cache[
                layer_idx, 0, physical_block, :, offset : offset + tokens_to_write, :
            ] = k[:, tokens_written : tokens_written + tokens_to_write, :]

            self.cache.cache[
                layer_idx, 1, physical_block, :, offset : offset + tokens_to_write, :
            ] = v[:, tokens_written : tokens_written + tokens_to_write, :]

            tokens_written += tokens_to_write
            current_len += tokens_to_write

        self.layer_seq_lens[layer_idx] = current_len


class StaticKVCache:
    """
    Zero-allocation, contiguous / ring-buffer KV-Cache for ultra-fast Apple Silicon inference.
    Pre-allocates a single unified memory buffer for all layers to eliminate per-token
    `torch.cat()` allocations and memory fragmentation during autoregressive decoding.
    Supports both monotonic growth and sliding-window ring-buffer eviction.
    """

    def __init__(
        self,
        max_batch_size: int,
        max_seq_len: int,
        n_layer: int,
        n_kv_head: int,
        head_dim: int,
        dtype: torch.dtype = torch.float16,
        device: torch.device = torch.device("mps"),
        ring_buffer: bool = False,
    ):
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.n_layer = n_layer
        self.n_kv_head = n_kv_head
        self.head_dim = head_dim
        self.ring_buffer = ring_buffer

        # Shape: [n_layer, 2 (K, V), max_batch_size, n_kv_head, max_seq_len, head_dim]
        self.cache = torch.empty(
            (n_layer, 2, max_batch_size, n_kv_head, max_seq_len, head_dim),
            dtype=dtype,
            device=device,
        )
        self.k_caches = [self.cache[i, 0] for i in range(n_layer)]
        self.v_caches = [self.cache[i, 1] for i in range(n_layer)]
        self.layer_seq_lens = [0] * n_layer
        self.total_seen_tokens = [0] * n_layer

    @property
    def seq_len(self) -> int:
        return self.total_seen_tokens[0]

    def reset(self):
        for i in range(self.n_layer):
            self.layer_seq_lens[i] = 0
            self.total_seen_tokens[i] = 0

    def update(
        self, layer_idx: int, k: torch.Tensor, v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        In-place update of K, V of shape [B, n_kv_head, T, head_dim].
        Returns zero-copy views of active K, V of shape [B, n_kv_head, active_len, head_dim].
        """
        B, _, T, _ = k.shape
        seen = self.total_seen_tokens[layer_idx]
        k_buf = self.k_caches[layer_idx]
        v_buf = self.v_caches[layer_idx]

        if not self.ring_buffer and seen + T <= self.max_seq_len:
            end_pos = seen + T
            if B == self.max_batch_size:
                k_buf[:, :, seen:end_pos, :] = k
                v_buf[:, :, seen:end_pos, :] = v
                self.layer_seq_lens[layer_idx] = end_pos
                self.total_seen_tokens[layer_idx] = end_pos
                return k_buf[:, :, :end_pos, :], v_buf[:, :, :end_pos, :]
            k_buf[:B, :, seen:end_pos, :] = k
            v_buf[:B, :, seen:end_pos, :] = v
            self.layer_seq_lens[layer_idx] = end_pos
            self.total_seen_tokens[layer_idx] = end_pos
            return k_buf[:B, :, :end_pos, :], v_buf[:B, :, :end_pos, :]
        else:
            # Ring-buffer wrap-around mode (O(1) bounded memory)
            if T >= self.max_seq_len:
                k_buf[:B, :, :, :] = k[:, :, -self.max_seq_len :, :]
                v_buf[:B, :, :, :] = v[:, :, -self.max_seq_len :, :]
            elif T == 1 and B == self.max_batch_size:
                write_pos = seen % self.max_seq_len
                k_buf[:, :, write_pos : write_pos + 1, :] = k
                v_buf[:, :, write_pos : write_pos + 1, :] = v
            else:
                write_pos = seen % self.max_seq_len
                first_chunk = min(T, self.max_seq_len - write_pos)
                k_buf[:B, :, write_pos : write_pos + first_chunk, :] = k[
                    :, :, :first_chunk, :
                ]
                v_buf[:B, :, write_pos : write_pos + first_chunk, :] = v[
                    :, :, :first_chunk, :
                ]
                rem = T - first_chunk
                if rem > 0:
                    k_buf[:B, :, :rem, :] = k[:, :, first_chunk:, :]
                    v_buf[:B, :, :rem, :] = v[:, :, first_chunk:, :]

            new_seen = seen + T
            active_len = min(new_seen, self.max_seq_len)
            self.layer_seq_lens[layer_idx] = active_len
            self.total_seen_tokens[layer_idx] = new_seen
            if B == self.max_batch_size:
                return k_buf[:, :, :active_len, :], v_buf[:, :, :active_len, :]
            return k_buf[:B, :, :active_len, :], v_buf[:B, :, :active_len, :]
