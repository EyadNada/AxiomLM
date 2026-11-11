import torch


class PagedKVCache:
    """
    [DEPRECATED for Production Inference]
    vLLM-style block table memory manager for KV-Cache.
    Eliminates memory fragmentation by storing KV states in non-contiguous memory blocks.

    NOTE (MLSys Architectural Warning):
    This class is preserved strictly as an unoptimized reference baseline. Reconstructing
    non-contiguous blocks via `get_reconstructed_cache()` invokes dynamic `torch.cat()`
    allocations on every decode step, triggering Unified Memory allocator stalls.
    For zero-allocation, bare-metal Apple Silicon inference, use `StaticKVCache` instead.
    """

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        n_layer: int,
        n_kv_head: int,
        head_dim: int,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
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
        self.block_table: list[int] = []
        self.layer_seq_lens = {}

    @property
    def seq_len(self):
        return max(self.layer_seq_lens.values()) if self.layer_seq_lens else 0

    def get_reconstructed_cache(
        self, layer_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        device: torch.device | str | None = None,
        ring_buffer: bool = False,
        cache_dtype: str = "fp16",
    ):
        if device is None:
            from axiomlm.kernels import is_mps_available

            device = torch.device("mps" if is_mps_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)

        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.n_layer = n_layer
        self.n_kv_head = n_kv_head
        self.head_dim = head_dim
        self.ring_buffer = ring_buffer
        self.cache_dtype = cache_dtype.lower()
        self.num_groups = max(1, head_dim // 64)

        if self.cache_dtype == "fp8":
            # FP8 E4M3 (1 byte per element) + scale (half per 64-element group)
            self.cache = torch.zeros(
                (n_layer, 2, max_batch_size, n_kv_head, max_seq_len, head_dim),
                dtype=torch.uint8,
                device=device,
            )
            self.scales = torch.zeros(
                (n_layer, 2, max_batch_size, n_kv_head, max_seq_len, self.num_groups),
                dtype=torch.float16,
                device=device,
            )
            self.k_scales = [self.scales[i, 0] for i in range(n_layer)]
            self.v_scales = [self.scales[i, 1] for i in range(n_layer)]
        elif self.cache_dtype == "int4":
            # Symmetric INT4 (0.5 bytes per element, packed 2 nibbles/byte) + scale
            self.cache = torch.zeros(
                (n_layer, 2, max_batch_size, n_kv_head, max_seq_len, head_dim // 2),
                dtype=torch.uint8,
                device=device,
            )
            self.scales = torch.zeros(
                (n_layer, 2, max_batch_size, n_kv_head, max_seq_len, self.num_groups),
                dtype=torch.float16,
                device=device,
            )
            self.k_scales = [self.scales[i, 0] for i in range(n_layer)]
            self.v_scales = [self.scales[i, 1] for i in range(n_layer)]
        else:
            # Native FP16 (2 bytes per element)
            self.cache = torch.empty(
                (n_layer, 2, max_batch_size, n_kv_head, max_seq_len, head_dim),
                dtype=dtype,
                device=device,
            )
            self.scales = None
            self.k_scales = None
            self.v_scales = None

        self.k_caches = [self.cache[i, 0] for i in range(n_layer)]
        self.v_caches = [self.cache[i, 1] for i in range(n_layer)]
        self.layer_seq_lens = [0] * n_layer
        self.total_seen_tokens = [0] * n_layer

    def get_memory_bytes(self) -> int:
        """Returns exact physical memory footprint allocated by this cache in bytes."""
        total = self.cache.numel() * self.cache.element_size()
        if self.scales is not None:
            total += self.scales.numel() * self.scales.element_size()
        return total

    @property
    def seq_len(self) -> int:
        return self.total_seen_tokens[0]

    def reset(self):
        for i in range(self.n_layer):
            self.layer_seq_lens[i] = 0
            self.total_seen_tokens[i] = 0

    def update(
        self, layer_idx: int, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        In-place update of K, V of shape [B, n_kv_head, T, head_dim].
        Returns active views of K, V. For quantized caches, returns high-precision
        dequantized views for downstream SDPA evaluation.
        """
        from axiomlm.kernels.quant import (
            dequantize_kv_fp8,
            dequantize_kv_int4,
            quantize_kv_fp8,
            quantize_kv_int4,
        )

        B, _, T, _ = k.shape
        seen = self.total_seen_tokens[layer_idx]
        k_buf = self.k_caches[layer_idx]
        v_buf = self.v_caches[layer_idx]

        if self.cache_dtype == "fp8":
            k_q, ks, v_q, vs = quantize_kv_fp8(k, v)
            k_s_buf = self.k_scales[layer_idx]
            v_s_buf = self.v_scales[layer_idx]
        elif self.cache_dtype == "int4":
            k_q, ks, v_q, vs = quantize_kv_int4(k, v)
            k_s_buf = self.k_scales[layer_idx]
            v_s_buf = self.v_scales[layer_idx]
        else:
            k_q, v_q = k, v
            k_s_buf, v_s_buf = None, None

        if not self.ring_buffer and seen + T <= self.max_seq_len:
            end_pos = seen + T
            k_buf[:B, :, seen:end_pos, :] = k_q
            v_buf[:B, :, seen:end_pos, :] = v_q
            if k_s_buf is not None and v_s_buf is not None:
                k_s_buf[:B, :, seen:end_pos, :] = ks
                v_s_buf[:B, :, seen:end_pos, :] = vs
            self.layer_seq_lens[layer_idx] = end_pos
            self.total_seen_tokens[layer_idx] = end_pos

            if self.cache_dtype == "fp8":
                return (
                    dequantize_kv_fp8(
                        k_buf[:B, :, :end_pos, :],
                        k_s_buf[:B, :, :end_pos, :],
                        self.head_dim,
                    ),
                    dequantize_kv_fp8(
                        v_buf[:B, :, :end_pos, :],
                        v_s_buf[:B, :, :end_pos, :],
                        self.head_dim,
                    ),
                )
            elif self.cache_dtype == "int4":
                return (
                    dequantize_kv_int4(
                        k_buf[:B, :, :end_pos, :],
                        k_s_buf[:B, :, :end_pos, :],
                        self.head_dim,
                    ),
                    dequantize_kv_int4(
                        v_buf[:B, :, :end_pos, :],
                        v_s_buf[:B, :, :end_pos, :],
                        self.head_dim,
                    ),
                )
            if B == self.max_batch_size:
                return k_buf[:, :, :end_pos, :], v_buf[:, :, :end_pos, :]
            return k_buf[:B, :, :end_pos, :], v_buf[:B, :, :end_pos, :]
        else:
            # Ring-buffer wrap-around mode (O(1) bounded memory)
            if T >= self.max_seq_len:
                k_buf[:B, :, :, :] = k_q[:, :, -self.max_seq_len :, :]
                v_buf[:B, :, :, :] = v_q[:, :, -self.max_seq_len :, :]
                if k_s_buf is not None and v_s_buf is not None:
                    k_s_buf[:B, :, :, :] = ks[:, :, -self.max_seq_len :, :]
                    v_s_buf[:B, :, :, :] = vs[:, :, -self.max_seq_len :, :]
            elif T == 1 and B == self.max_batch_size:
                write_pos = seen % self.max_seq_len
                k_buf[:, :, write_pos : write_pos + 1, :] = k_q
                v_buf[:, :, write_pos : write_pos + 1, :] = v_q
                if k_s_buf is not None and v_s_buf is not None:
                    k_s_buf[:, :, write_pos : write_pos + 1, :] = ks
                    v_s_buf[:, :, write_pos : write_pos + 1, :] = vs
            else:
                write_pos = seen % self.max_seq_len
                first_chunk = min(T, self.max_seq_len - write_pos)
                k_buf[:B, :, write_pos : write_pos + first_chunk, :] = k_q[
                    :, :, :first_chunk, :
                ]
                v_buf[:B, :, write_pos : write_pos + first_chunk, :] = v_q[
                    :, :, :first_chunk, :
                ]
                if k_s_buf is not None and v_s_buf is not None:
                    k_s_buf[:B, :, write_pos : write_pos + first_chunk, :] = ks[
                        :, :, :first_chunk, :
                    ]
                    v_s_buf[:B, :, write_pos : write_pos + first_chunk, :] = vs[
                        :, :, :first_chunk, :
                    ]
                rem = T - first_chunk
                if rem > 0:
                    k_buf[:B, :, :rem, :] = k_q[:, :, first_chunk:, :]
                    v_buf[:B, :, :rem, :] = v_q[:, :, first_chunk:, :]
                    if k_s_buf is not None and v_s_buf is not None:
                        k_s_buf[:B, :, :rem, :] = ks[:, :, first_chunk:, :]
                        v_s_buf[:B, :, :rem, :] = vs[:, :, first_chunk:, :]

            new_seen = seen + T
            active_len = min(new_seen, self.max_seq_len)
            self.layer_seq_lens[layer_idx] = active_len
            self.total_seen_tokens[layer_idx] = new_seen

            if self.cache_dtype == "fp8":
                return (
                    dequantize_kv_fp8(
                        k_buf[:B, :, :active_len, :],
                        k_s_buf[:B, :, :active_len, :],
                        self.head_dim,
                    ),
                    dequantize_kv_fp8(
                        v_buf[:B, :, :active_len, :],
                        v_s_buf[:B, :, :active_len, :],
                        self.head_dim,
                    ),
                )
            elif self.cache_dtype == "int4":
                return (
                    dequantize_kv_int4(
                        k_buf[:B, :, :active_len, :],
                        k_s_buf[:B, :, :active_len, :],
                        self.head_dim,
                    ),
                    dequantize_kv_int4(
                        v_buf[:B, :, :active_len, :],
                        v_s_buf[:B, :, :active_len, :],
                        self.head_dim,
                    ),
                )
            if B == self.max_batch_size:
                return k_buf[:, :, :active_len, :], v_buf[:, :, :active_len, :]
            return k_buf[:B, :, :active_len, :], v_buf[:B, :, :active_len, :]

    def get_quantized_views(
        self, layer_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """
        Returns raw active memory views (k_buf, v_buf, k_scales, v_scales)
        without allocating or reconstructing dynamic dequantized FP16 tensors.
        """
        k_buf = self.k_caches[layer_idx]
        v_buf = self.v_caches[layer_idx]
        k_s = self.k_scales[layer_idx] if self.scales is not None else None
        v_s = self.v_scales[layer_idx] if self.scales is not None else None
        return k_buf, v_buf, k_s, v_s

    def compact_tree_path(
        self,
        write_pos: int,
        accepted_indices: list[int],
        new_total_seen: int | None = None,
    ) -> None:
        """
        In-place topological KV-cache compaction after SpecInfer / Medusa tree verification.
        Moves KV vectors of accepted tree nodes `(write_pos + accepted_indices[m]) % max_seq_len`
        into contiguous slots `(write_pos + m) % max_seq_len` across all layers in a single
        race-free pass, then updates `total_seen_tokens` and `layer_seq_lens`.
        """
        num_accepted = len(accepted_indices)
        if num_accepted > 0:
            already_compact = all(accepted_indices[m] == m for m in range(num_accepted))
            if not already_compact:
                from axiomlm.kernels.ops import _METAL_MOD

                if (
                    self.cache.device.type == "mps"
                    and self.cache.is_contiguous()
                    and _METAL_MOD is not None
                ):
                    if self.cache_dtype == "fp16" and hasattr(
                        _METAL_MOD, "compact_tree_kv_mps"
                    ):
                        _METAL_MOD.compact_tree_kv_mps(
                            self.cache,
                            int(write_pos),
                            [int(x) for x in accepted_indices],
                        )
                    elif (
                        self.cache_dtype in ("fp8", "int4")
                        and self.scales is not None
                        and hasattr(_METAL_MOD, "compact_tree_kv_quant_mps")
                    ):
                        _METAL_MOD.compact_tree_kv_quant_mps(
                            self.cache,
                            self.scales,
                            int(write_pos),
                            [int(x) for x in accepted_indices],
                            self.cache_dtype,
                        )
                    else:
                        src_slots = [
                            (write_pos + int(idx)) % self.max_seq_len
                            for idx in accepted_indices
                        ]
                        dst_slots = [
                            (write_pos + m) % self.max_seq_len
                            for m in range(num_accepted)
                        ]
                        src_t = torch.tensor(
                            src_slots, dtype=torch.long, device=self.cache.device
                        )
                        dst_t = torch.tensor(
                            dst_slots, dtype=torch.long, device=self.cache.device
                        )
                        staged = self.cache.index_select(4, src_t).clone()
                        self.cache.index_copy_(4, dst_t, staged)
                        if self.scales is not None:
                            staged_s = self.scales.index_select(4, src_t).clone()
                            self.scales.index_copy_(4, dst_t, staged_s)
                else:
                    src_slots = [
                        (write_pos + int(idx)) % self.max_seq_len
                        for idx in accepted_indices
                    ]
                    dst_slots = [
                        (write_pos + m) % self.max_seq_len for m in range(num_accepted)
                    ]
                    src_t = torch.tensor(
                        src_slots, dtype=torch.long, device=self.cache.device
                    )
                    dst_t = torch.tensor(
                        dst_slots, dtype=torch.long, device=self.cache.device
                    )
                    staged = self.cache.index_select(4, src_t).clone()
                    self.cache.index_copy_(4, dst_t, staged)
                    if self.scales is not None:
                        staged_s = self.scales.index_select(4, src_t).clone()
                        self.scales.index_copy_(4, dst_t, staged_s)

        if new_total_seen is not None:
            active = min(new_total_seen, self.max_seq_len)
            self.total_seen_tokens[:] = [new_total_seen] * self.n_layer
            self.layer_seq_lens[:] = [active] * self.n_layer
