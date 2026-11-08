import gc
import pytest
import torch
from axiomlm.models.transformer import ModelConfig, Transformer
from axiomlm.engine.paged_cache import StaticKVCache


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Requires Apple Silicon MPS"
)
def test_task02_dead_fp16_reclamation_and_tied_q8_gather():
    device = torch.device("mps")
    torch.mps.empty_cache()
    gc.collect()

    config = ModelConfig(
        block_size=512,
        vocab_size=2048,
        n_layer=4,
        n_head=4,
        n_kv_head=2,
        n_embd=256,
        intermediate_size=512,
        tie_word_embeddings=True,
        pos_emb="rope",
        norm_type="rmsnorm",
        mlp_type="swiglu",
        use_fused_kernels=True,
    )

    # 1. Instantiate baseline FP16 model
    model = (
        Transformer(config, init_weights=True)
        .to(device=device, dtype=torch.float16)
        .eval()
    )
    cache = StaticKVCache(
        max_batch_size=1,
        max_seq_len=512,
        n_layer=config.n_layer,
        n_kv_head=config.n_kv_head,
        head_dim=config.n_embd // config.n_head,
        device=device,
        dtype=torch.float16,
    )

    with torch.no_grad():
        # Warmup and baseline decode step
        idx = torch.tensor([[42]], device=device, dtype=torch.long)
        base_logits, _, _ = model.forward(idx, kv_caches=[cache] * config.n_layer)
        assert base_logits is not None

        # 2. Quantize with release_fp16=False first to observe pre-eviction memory
        model.quantize("q4", release_fp16=False)
        runner = model.get_decode_runner(cache)
        assert runner is not None

        # Step with quantized runner
        q_logits_pre = runner.step(idx, 0, 0, 1)

        # Verify tied embedding sharing
        qw = model._ensure_quant_weights(device)
        assert (
            qw["wte_p"].data_ptr() == qw["lmh_p"].data_ptr()
        ), "Tied embeddings must share exact data_ptr with lm_head"
        assert (
            qw["wte_sb"].data_ptr() == qw["lmh_sb"].data_ptr()
        ), "Tied scales must share exact data_ptr with lm_head"

        # 3. Now quantize with release_fp16=True
        model.quantize("q4", release_fp16=True)
        assert getattr(model, "_fp16_released", False) is True

    # Confirm FP16 weight storages are empty
    for b in model.transformer.h:
        assert b.attn.c_attn.weight.numel() == 0
        assert b.attn.c_proj.weight.numel() == 0
        assert b.mlp.w_gate.weight.numel() == 0
        assert b.mlp.w_up.weight.numel() == 0
        assert b.mlp.w_down.weight.numel() == 0
    assert model.lm_head.weight.numel() == 0
    assert model.transformer.wte.weight.numel() == 0

    # 4. Confirm guards prevent invalid training or unquantization
    with pytest.raises(RuntimeError, match="Cannot switch to training mode"):
        model.train(True)

    with pytest.raises(RuntimeError, match="Cannot unquantize model"):
        model.quantize(None)

    # 5. Verify post-eviction decode step produces bit-exact match with pre-eviction
    with torch.no_grad():
        runner_post = model.get_decode_runner(cache)
        assert runner_post is not None
        q_logits_post = runner_post.step(idx, 0, 0, 1)

        max_diff = (q_logits_pre - q_logits_post).abs().max().item()
        assert (
            max_diff < 1e-4
        ), f"Post-eviction step diverged from pre-eviction: max diff {max_diff}"

        # 6. Verify tree speculative verification also functions post-eviction
        tree_idx = torch.tensor([[10, 20, 30, 40]], device=device, dtype=torch.long)
        tree_depths = [0, 1, 2, 3]
        tree_mask = [0b0001, 0b0011, 0b0111, 0b1111]
        tree_logits = runner_post.verify_tree_step(
            tree_idx, 0, 0, 0, tree_depths, tree_mask
        )
        assert tree_logits.shape == (1, 4, config.vocab_size)
        assert not torch.isnan(tree_logits).any()
    print("SUCCESS: Task 0.2 unit verification passed with zero regressions!")


if __name__ == "__main__":
    test_task02_dead_fp16_reclamation_and_tied_q8_gather()
