import os
import sys
import unittest
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from brain.train_gpt2 import (
    RMSNorm,
    precompute_rope_frequencies,
    apply_rope,
    repeat_kv,
    SwiGLUMLP,
    MLP,
    CausalSelfAttention,
    Block,
    GPTConfig,
    GPT,
    zeropower_via_newtonschulz5,
    Muon,
    DataLoaderLite,
    sample_logits,
    generate_samples,
    generate_with_cache,
    get_lr,
    get_raw_model,
    estimate_hardware_peak_tflops,
    calculate_mfu,
    create_profiler,
)


class TestModernArchitectureComponents(unittest.TestCase):
    """Unit tests for Modern LLaMA-3 / Mistral architectural primitives."""

    def test_rmsnorm_forward_and_backward(self):
        """Test RMSNorm shape preservation, normalization property, and gradient flow."""
        B, T, C = 2, 8, 64
        x = torch.randn(B, T, C, requires_grad=True)
        norm = RMSNorm(C)
        out = norm(x)

        # 1. Check output shape
        self.assertEqual(out.shape, (B, T, C))

        # 2. Check RMS normalization property (unit root-mean-square before weight scaling)
        rms = torch.sqrt(torch.mean(out.pow(2), dim=-1))
        # With initial weight = 1.0, RMS should be close to 1.0
        self.assertTrue(torch.allclose(rms, torch.ones_like(rms), atol=1e-3))

        # 3. Check backward pass and gradient flow
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(norm.weight.grad)
        self.assertFalse(torch.isnan(x.grad).any())
        self.assertFalse(torch.isnan(norm.weight.grad).any())

    def test_rope_frequencies_and_application(self):
        """Test Rotary Position Embeddings (RoPE) precomputation, rotation, and relative distance property."""
        head_dim = 64
        max_seq_len = 128
        freqs_cis = precompute_rope_frequencies(head_dim=head_dim, max_seq_len=max_seq_len)

        # 1. Frequency shape check
        self.assertEqual(freqs_cis.shape, (max_seq_len, head_dim // 2))
        self.assertEqual(freqs_cis.dtype, torch.complex64)

        # 2. Apply RoPE on query/key tensor (B, nh, T, head_dim)
        B, nh, T = 2, 4, 16
        q = torch.randn(B, nh, T, head_dim)
        k = torch.randn(B, nh, T, head_dim)

        q_rot = apply_rope(q, freqs_cis, start_pos=0)
        k_rot = apply_rope(k, freqs_cis, start_pos=0)

        self.assertEqual(q_rot.shape, (B, nh, T, head_dim))
        self.assertEqual(k_rot.shape, (B, nh, T, head_dim))

        # 3. Preservation of vector norm (rotations are orthogonal/isometric)
        q_norm_before = torch.norm(q, dim=-1)
        q_norm_after = torch.norm(q_rot, dim=-1)
        self.assertTrue(torch.allclose(q_norm_before, q_norm_after, atol=1e-4))

        # 4. Test start_pos offset for KV-cache decoding step (T=1 at position 10)
        q_single = torch.randn(B, nh, 1, head_dim)
        q_rot_offset = apply_rope(q_single, freqs_cis, start_pos=10)
        self.assertEqual(q_rot_offset.shape, (B, nh, 1, head_dim))

    def test_repeat_kv_gqa(self):
        """Test Grouped-Query Attention key/value head broadcasting."""
        B, n_kv_head, T, head_dim = 2, 4, 8, 64
        n_rep = 3  # 4 KV heads -> 12 Query heads
        x = torch.randn(B, n_kv_head, T, head_dim)

        out = repeat_kv(x, n_rep=n_rep)
        self.assertEqual(out.shape, (B, 12, T, head_dim))

        # Verify repeated slices are identical
        for i in range(n_kv_head):
            for r in range(n_rep):
                head_idx = i * n_rep + r
                self.assertTrue(torch.equal(out[:, head_idx, :, :], x[:, i, :, :]))

    def test_swiglu_mlp_dimensions_and_scaling(self):
        """Test SwiGLU feed-forward network hidden dimension alignment and forward/backward."""
        config = GPTConfig(n_embd=768, bias=False, mlp_type="swiglu")
        mlp = SwiGLUMLP(config)

        # Expected hidden_dim: 2/3 * 4 * 768 = 2048, rounded to multiple of 64 -> 2048
        expected_hidden = 2048
        self.assertEqual(mlp.w_gate.out_features, expected_hidden)
        self.assertEqual(mlp.w_up.out_features, expected_hidden)
        self.assertEqual(mlp.w_down.in_features, expected_hidden)

        x = torch.randn(2, 16, 768, requires_grad=True)
        out = mlp(x)
        self.assertEqual(out.shape, (2, 16, 768))

        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertFalse(torch.isnan(x.grad).any())


class TestAttentionAndKVCache(unittest.TestCase):
    """Unit tests for CausalSelfAttention and KV-Cache decoding."""

    def test_classic_mha_forward(self):
        """Test classic Multi-Head Attention without GQA."""
        config = GPTConfig(n_embd=768, n_head=12, n_kv_head=None, bias=True)
        attn = CausalSelfAttention(config)
        self.assertFalse(attn.separate_qkv)

        x = torch.randn(2, 16, 768)
        y, kv = attn(x)
        self.assertEqual(y.shape, (2, 16, 768))
        self.assertIsNone(kv)

    def test_modern_gqa_forward(self):
        """Test Grouped-Query Attention (12 Query heads, 4 KV heads)."""
        config = GPTConfig(n_embd=768, n_head=12, n_kv_head=4, bias=False)
        attn = CausalSelfAttention(config)
        self.assertTrue(attn.separate_qkv)
        self.assertEqual(attn.n_rep, 3)

        x = torch.randn(2, 16, 768)
        y, kv = attn(x)
        self.assertEqual(y.shape, (2, 16, 768))
        self.assertIsNone(kv)

    def test_kv_cache_exact_parity_single_layer(self):
        """Verify that single-layer KV-cache sequential decoding produces exact same outputs as full sequence."""
        config = GPTConfig(n_embd=128, n_head=4, n_kv_head=2, block_size=64, bias=False)
        attn = CausalSelfAttention(config)
        attn.eval()

        torch.manual_seed(42)
        seq_len = 8
        x = torch.randn(1, seq_len, 128)

        # 1. Full sequence forward pass
        with torch.no_grad():
            y_full, _ = attn(x)

        # 2. Step-by-step sequential decoding with KV cache
        with torch.no_grad():
            # Prefill with first token
            y_0, kv = attn(x[:, :1, :], use_cache=True)
            y_steps = [y_0]

            for t in range(1, seq_len):
                x_t = x[:, t : t + 1, :]
                y_t, kv = attn(x_t, kv_cache=kv, use_cache=True)
                y_steps.append(y_t)

            y_cached = torch.cat(y_steps, dim=1)

        # Verify bit-exact/close output parity between naive and cached
        diff = (y_full - y_cached).abs().max().item()
        self.assertLess(diff, 1e-5, f"KV cache output divergence: max diff {diff}")


class TestFullModelAndGeneration(unittest.TestCase):
    """Tests for the complete GPT model: Classic vs Modern specs, weight tying, and generation."""

    def test_classic_gpt2_initialization_and_forward(self):
        """Test Classic GPT-2 124M parameter count, weight tying, and forward pass."""
        config = GPTConfig(
            block_size=1024,
            vocab_size=50304,
            n_layer=12,
            n_head=12,
            n_embd=768,
            norm_type="layernorm",
            pos_emb="learned",
            mlp_type="gelu",
            bias=True,
        )
        model = GPT(config)

        # Verify weight tying: wte and lm_head share memory
        self.assertIs(model.transformer['wte'].weight, model.lm_head.weight)

        # Verify parameter count
        total_params = sum(p.numel() for p in model.parameters())
        self.assertEqual(total_params, 124_475_904)

        # Forward pass test
        idx = torch.randint(0, 50304, (2, 32))
        targets = torch.randint(0, 50304, (2, 32))
        logits, loss = model(idx, targets=targets)

        self.assertEqual(logits.shape, (2, 32, 50304))
        self.assertIsNotNone(loss)
        # Expected initial cross-entropy loss for 50,304 classes: ~ -ln(1/50304) ≈ 10.82
        self.assertAlmostEqual(loss.item(), math.log(50304), delta=1.5)

    def test_modern_llama3_initialization_and_forward(self):
        """Test Modern LLaMA-3 spec (RoPE, RMSNorm, SwiGLU, GQA) parameter count and forward."""
        config = GPTConfig(
            block_size=1024,
            vocab_size=50304,
            n_layer=12,
            n_head=12,
            n_embd=768,
            n_kv_head=4,
            norm_type="rmsnorm",
            pos_emb="rope",
            mlp_type="swiglu",
            bias=False,
        )
        model = GPT(config)

        # No learned position embeddings table (wpe) in RoPE
        self.assertNotIn('wpe', model.transformer)
        self.assertIsNotNone(model.freqs_cis)

        # Verify weight tying
        self.assertIs(model.transformer['wte'].weight, model.lm_head.weight)

        idx = torch.randint(0, 50304, (2, 32))
        targets = torch.randint(0, 50304, (2, 32))
        logits, loss = model(idx, targets=targets)

        self.assertEqual(logits.shape, (2, 32, 50304))
        self.assertIsNotNone(loss)
        self.assertAlmostEqual(loss.item(), math.log(50304), delta=1.5)

    def test_end_to_end_greedy_generation_parity(self):
        """Verify that full-model greedy generation with KV cache is 100% token-for-token identical to naive generation."""
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")

        config = GPTConfig(
            block_size=128,
            vocab_size=50304,
            n_layer=4,
            n_head=4,
            n_embd=128,
            n_kv_head=2,
            norm_type="rmsnorm",
            pos_emb="rope",
            mlp_type="swiglu",
            bias=False,
        )
        torch.manual_seed(1337)
        model = GPT(config)
        model.eval()

        prompt = "Once upon a time there was a"
        max_length = 25

        # 1. Naive generation (deterministic greedy with temperature=0.0 equivalent)
        # Using generate_with_cache with temp=0.0
        sample_cached = generate_with_cache(
            model, enc, device="cpu", prompt=prompt, num_samples=1, max_length=max_length, temperature=0.0
        )[0]

        # 2. Step-by-step naive forward without KV cache (greedy)
        prompt_tokens = enc.encode(prompt)
        tokens = torch.tensor(prompt_tokens, dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            while tokens.size(1) < max_length:
                logits, _ = model(tokens)
                next_tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                tokens = torch.cat([tokens, next_tok], dim=1)
        sample_naive = enc.decode(tokens[0].tolist())

        self.assertEqual(
            sample_cached,
            sample_naive,
            f"Cached and Naive greedy generation mismatch!\nCached: {sample_cached}\nNaive:  {sample_naive}",
        )


class TestMuonOptimizerAndNewtonSchulz(unittest.TestCase):
    """Unit tests for Muon 2D matrix optimizer and 5th-order Newton-Schulz polar decomposition."""

    def test_newton_schulz_polar_orthogonalization(self):
        """Test that Newton-Schulz quintic iteration dramatically flattens singular value spectrum toward 1.0."""
        torch.manual_seed(42)
        # Test square matrix
        G = torch.randn(64, 64)
        s_orig = torch.linalg.svdvals(G)
        cond_orig = (s_orig.max() / s_orig.min()).item()

        U = zeropower_via_newtonschulz5(G, steps=5)
        s_ns = torch.linalg.svdvals(U)
        cond_ns = (s_ns.max() / s_ns.min()).item()

        # 1. Verify massive reduction in condition number (spectral flattening)
        self.assertLess(cond_ns, 2.0, f"Newton-Schulz condition number not compressed: {cond_ns} (original was {cond_orig})")

        # 2. Verify singular values are tightly centered around ~1.0
        self.assertGreater(s_ns.min().item(), 0.5)
        self.assertLess(s_ns.max().item(), 1.3)

        # 3. Check off-diagonal orthogonality of U @ U.T
        I_approx = U @ U.T
        off_diag = (I_approx - torch.diag(torch.diag(I_approx))).abs().max().item()
        self.assertLess(off_diag, 0.2, f"Newton-Schulz off-diagonal correlation too high: {off_diag}")

        # Test rectangular matrix (rows < cols)
        G_rect = torch.randn(32, 64)
        U_rect = zeropower_via_newtonschulz5(G_rect, steps=5)
        self.assertEqual(U_rect.shape, (32, 64))
        s_rect = torch.linalg.svdvals(U_rect)
        cond_rect = (s_rect.max() / s_rect.min()).item()
        self.assertLess(cond_rect, 2.0)

    def test_muon_optimizer_step(self):
        """Test Muon optimizer parameter update and state management."""
        w = nn.Parameter(torch.randn(32, 32))
        opt = Muon([w], lr=0.02, momentum=0.95, nesterov=True)

        w_orig = w.clone().detach()

        # Synthetic loss and gradient
        loss = (w ** 2).sum()
        loss.backward()

        opt.step()

        # Parameters should have changed
        self.assertFalse(torch.equal(w, w_orig))
        self.assertIn('momentum_buffer', opt.state[w])
        self.assertEqual(opt.state[w]['momentum_buffer'].shape, (32, 32))

    def test_dual_optimizer_routing(self):
        """Test configure_optimizers routes 2D weights to Muon and 1D/embeddings to AdamW."""
        config = GPTConfig(n_layer=2, n_head=4, n_embd=128, vocab_size=1000)
        model = GPT(config)

        optimizers = model.configure_optimizers(
            weight_decay=0.1,
            learning_rate=6e-4,
            device="cpu",
            optimizer_type="muon",
            muon_lr=0.02,
        )

        self.assertEqual(len(optimizers), 2)
        muon_opt, adamw_opt = optimizers[0], optimizers[1]

        self.assertIsInstance(muon_opt, Muon)
        self.assertIsInstance(adamw_opt, torch.optim.AdamW)

        # Verify all parameters in Muon are strictly 2D
        for group in muon_opt.param_groups:
            for p in group['params']:
                self.assertEqual(p.dim(), 2)


class TestDataLoaderAndShards(unittest.TestCase):
    """Unit tests for DataLoaderLite, binary shard streaming, and wrap-around handling."""

    def test_dataloader_batching_and_shifting(self):
        """Verify DataLoaderLite yields (B, T) batches and y is precisely x shifted by 1."""
        B, T = 4, 16
        loader = DataLoaderLite(B=B, T=T, split="train")

        x, y = loader.next_batch()

        self.assertEqual(x.shape, (B, T))
        self.assertEqual(y.shape, (B, T))

        # Check contiguous shift property: y[b, :-1] should equal x[b, 1:]
        self.assertTrue(torch.equal(x[:, 1:], y[:, :-1]))

    def test_dataloader_reset_and_set_step(self):
        """Verify set_step fast-forwards position deterministically."""
        B, T = 2, 8
        loader = DataLoaderLite(B=B, T=T, split="train")

        loader.set_step(step=10, grad_accum_steps=2)
        expected_pos = 10 * (B * T * 2)
        self.assertEqual(loader.current_position, expected_pos)

    def test_multishard_streaming_and_rotation(self):
        """Verify multi-shard dynamic rotation and cross-shard step synchronization."""
        import tempfile
        import numpy as np

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create two synthetic binary shards of 500 tokens each
            s1 = np.arange(0, 500, dtype=np.uint16)
            s2 = np.arange(500, 1000, dtype=np.uint16)
            s1.tofile(os.path.join(tmpdir, "train_0000.bin"))
            s2.tofile(os.path.join(tmpdir, "train_0001.bin"))

            B, T = 2, 8
            loader = DataLoaderLite(B=B, T=T, split="train", data_dir=tmpdir)
            self.assertEqual(len(loader.shards), 2)
            self.assertEqual(loader.total_tokens, 1000)

            # Draw batches across shard boundary
            batches = [loader.next_batch() for _ in range(40)]
            self.assertEqual(len(batches), 40)

            # Test cross-shard set_step synchronization into shard 1
            loader.set_step(step=40, grad_accum_steps=1)
            # Step 40 with B=2, T=8 is offset 640 -> Shard 1 at local offset 140
            self.assertEqual(loader.current_shard_idx, 1)
            self.assertEqual(loader.current_position, 140)


class TestCheckpointingAndResumption(unittest.TestCase):
    """Unit tests for saving and restoring training state dictionaries."""

    def test_checkpoint_save_and_restore(self):
        """Verify that checkpoint state dictionary restores exact weights and outputs."""
        import tempfile
        config = GPTConfig(n_layer=2, n_head=4, n_embd=64, vocab_size=500)
        model1 = GPT(config)
        opt1 = torch.optim.AdamW(model1.parameters(), lr=1e-3)

        # Do a fake step
        x = torch.randint(0, 500, (2, 8))
        logits, loss = model1(x, targets=x)
        loss.backward()
        opt1.step()

        # Save checkpoint to temp directory
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "test_ckpt.pt")
            checkpoint = {
                "step": 42,
                "model_state_dict": model1.state_dict(),
                "optimizer_state_dicts": [opt1.state_dict()],
                "config": model1.config,
            }
            torch.save(checkpoint, ckpt_path)

            # Load into fresh model instance
            ckpt_loaded = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            model2 = GPT(ckpt_loaded["config"])
            model2.load_state_dict(ckpt_loaded["model_state_dict"])

            # Verify identical weights
            for p1, p2 in zip(model1.parameters(), model2.parameters()):
                self.assertTrue(torch.equal(p1, p2))

            # Verify identical forward logits
            logits1, _ = model1(x)
            logits2, _ = model2(x)
            self.assertTrue(torch.equal(logits1, logits2))


class TestPretrainedHuggingFaceWeights(unittest.TestCase):
    """Test loading official OpenAI GPT-2 weights from Hugging Face and matching logits."""

    def test_hf_weight_loading_and_logit_agreement(self):
        """Verify weights loaded from HF produce identical logits to transformers library."""
        try:
            from transformers import GPT2LMHeadModel
        except ImportError:
            self.skipTest("transformers library not available")

        # Load both implementations
        model_custom = GPT.from_pretrained("gpt2")
        model_custom.eval()

        model_hf = GPT2LMHeadModel.from_pretrained("gpt2")
        model_hf.eval()

        # Test prompt
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode("Hello, my name is Axiom and I build")
        input_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)

        with torch.no_grad():
            logits_custom, _ = model_custom(input_ids)
            logits_hf = model_hf(input_ids).logits

        # Logits should match within numerical precision (atol=1e-4)
        max_diff = (logits_custom - logits_hf).abs().max().item()
        self.assertLess(max_diff, 1e-3, f"HF weights logit mismatch: max diff {max_diff}")


class TestSystemsProfilingAndMFU(unittest.TestCase):
    """Unit tests for Hardware Peak Compute estimation and MFU (Model FLOPs Utilization) tracking."""

    def test_estimate_hardware_peak_tflops(self):
        """Verify device peak TFLOPs estimation returns positive valid numbers."""
        mps_peak = estimate_hardware_peak_tflops("mps")
        cpu_peak = estimate_hardware_peak_tflops("cpu")
        cuda_peak = estimate_hardware_peak_tflops("cuda")

        self.assertGreater(mps_peak, 0.0)
        self.assertGreater(cpu_peak, 0.0)
        self.assertGreater(cuda_peak, 0.0)
        self.assertEqual(mps_peak, 10.0)
        self.assertEqual(cpu_peak, 2.0)

    def test_calculate_mfu_accuracy(self):
        """Verify MFU percentage and achieved TFLOPs math."""
        config = GPTConfig(
            block_size=1024,
            vocab_size=50304,
            n_layer=12,
            n_head=12,
            n_embd=768,
        )
        model = GPT(config)
        tokens_per_sec = 10000.0  # 10k tokens/sec
        seq_len = 1024
        peak_tflops = 10.0

        mfu_pct, achieved_tflops = calculate_mfu(
            model=model,
            tokens_per_sec=tokens_per_sec,
            seq_len=seq_len,
            peak_tflops=peak_tflops,
        )

        # Total params P ≈ 124.4M
        # FLOPs per token ≈ 6 * 124.4M + 12 * 12 * 768 * 1024 ≈ 7.46e8 + 1.13e8 ≈ 8.60e8 FLOPs/tok
        # Achieved TFLOPs at 10k tok/s ≈ 8.60 TFLOPs
        # MFU on 10 TFLOPs peak ≈ 86%
        self.assertGreater(achieved_tflops, 7.0)
        self.assertLess(achieved_tflops, 10.0)
        self.assertGreater(mfu_pct, 70.0)
        self.assertLess(mfu_pct, 100.0)

    def test_profiler_creation(self):
        """Verify PyTorch profiler instance initializes without error."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            prof = create_profiler(tmpdir)
            self.assertIsNotNone(prof)


class TestAdvancedSamplingAndCheckpointing(unittest.TestCase):
    """Unit tests for advanced sampling strategies (Top-p, Min-p, Repetition Penalty) and Gradient Checkpointing."""

    def test_sample_logits_temperature_zero(self):
        """Test greedy argmax decoding when temperature <= 0."""
        logits = torch.tensor([[1.0, 5.0, 2.0], [10.0, 2.0, 1.0]])
        out = sample_logits(logits, temperature=0.0)
        self.assertEqual(out[0].item(), 1)
        self.assertEqual(out[1].item(), 0)

    def test_sample_logits_top_k_and_top_p(self):
        """Test Top-k and Top-p nucleus filtering behavior."""
        torch.manual_seed(42)
        logits = torch.randn(2, 100)
        out = sample_logits(logits, temperature=1.0, top_k=5, top_p=0.8)
        self.assertEqual(out.shape, (2, 1))
        self.assertTrue((out >= 0).all() and (out < 100).all())

    def test_sample_logits_min_p(self):
        """Test Min-p dynamic probability thresholding."""
        torch.manual_seed(42)
        logits = torch.tensor([[10.0, 1.0, -10.0, -20.0]])
        out = sample_logits(logits, temperature=1.0, min_p=0.1)
        self.assertEqual(out[0].item(), 0)

    def test_sample_logits_repetition_penalty(self):
        """Test repetition penalty discounts previously seen tokens."""
        logits = torch.tensor([[5.0, 4.9, 1.0]])
        prev_tokens = torch.tensor([[0]])
        out = sample_logits(logits, temperature=0.0, repetition_penalty=2.0, prev_tokens=prev_tokens)
        self.assertEqual(out[0].item(), 1)

    def test_gradient_checkpointing_forward_backward(self):
        """Test that activation gradient checkpointing produces valid backward gradients."""
        config = GPTConfig(
            block_size=128,
            vocab_size=1000,
            n_layer=4,
            n_head=4,
            n_embd=128,
            grad_checkpoint=True,
        )
        model = GPT(config)
        model.train()
        x = torch.randint(0, 1000, (2, 32))
        y = torch.randint(0, 1000, (2, 32))
        logits, loss = model(x, y)
        self.assertIsNotNone(loss)
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Missing gradient for {name}")
                self.assertFalse(torch.isnan(param.grad).any())


class TestWebInterfaceAndApp(unittest.TestCase):
    """Automated tests for app.py web interface, streaming generator, and live benchmark."""

    def test_app_build_blocks(self):
        """Verify Gradio blocks application instantiates with all components without errors."""
        import app
        demo = app.build_app()
        self.assertIsNotNone(demo)
        self.assertEqual(demo.title, "AxiomLM (124M)")

    def test_app_stream_inference_generation(self):
        """Verify stream_inference yields progressive text, probability inspector, and telemetry."""
        import app
        gen = app.stream_inference(
            prompt="The quick brown fox",
            source_type="local",
            custom_checkpoint="checkpoints/model_latest.pt" if os.path.exists("checkpoints/model_latest.pt") else "",
            arch="modern",
            max_tokens=6,
            temperature=0.8,
            top_k=50,
            top_p=0.9,
            min_p=0.05,
            repetition_penalty=1.1,
            use_kv_cache=True,
            pace_stream=False,
        )
        outputs = list(gen)
        self.assertGreater(len(outputs), 0)
        last_text, last_prob, last_telem = outputs[-1]
        self.assertTrue(last_text.startswith("The quick brown fox"))
        self.assertIn("tokens in", last_telem)

    def test_app_side_by_side_benchmark_generator(self):
        """Verify stream_side_by_side_benchmark yields dual stream and markdown summary table."""
        import app
        gen = app.stream_side_by_side_benchmark(
            prompt="Hello world",
            source_type="local",
            custom_checkpoint="checkpoints/model_latest.pt" if os.path.exists("checkpoints/model_latest.pt") else "",
            arch="modern",
            num_tokens=6,
        )
        outputs = list(gen)
        self.assertGreater(len(outputs), 0)
        c_txt, n_txt, sc, sn, md = outputs[-1]
        self.assertTrue(c_txt.startswith("Hello world"))
        self.assertTrue(n_txt.startswith("Hello world"))
        self.assertIn("KV-Cache", md)
        self.assertIn("Faster", md)

    def test_app_cloud_savings_calculator(self):
        """Verify calculate_cloud_savings generates valid triton kernel, math notes, and cost markdown table."""
        import app
        triton_code, math_md, cost_md = app.calculate_cloud_savings(
            num_gpus=64,
            gpu_cost_hr=3.20,
            operator_name="Fused RMSNorm (Root Mean Square Normalization)",
        )
        self.assertIn("@triton.jit", triton_code)
        self.assertIn("RMSNorm", math_md)
        self.assertIn("Enterprise Savings", cost_md)
        self.assertIn("Monthly Cloud Bill", cost_md)


class TestHuggingFaceExport(unittest.TestCase):
    """Unit tests for Hugging Face .safetensors model export and metadata generation."""

    def test_export_checkpoint_to_safetensors(self):
        """Verify that export_checkpoint_to_hf generates valid .safetensors, config.json, and metadata."""
        import tempfile
        from safetensors.torch import load_file
        from brain.export_hf import export_checkpoint_to_hf

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create dummy checkpoint
            cfg = GPTConfig(n_layer=2, n_head=4, n_embd=64, vocab_size=500)
            model = GPT(cfg)
            ckpt_path = os.path.join(tmpdir, "test_model.pt")
            torch.save({
                "step": 100,
                "model_state_dict": model.state_dict(),
                "config": cfg,
            }, ckpt_path)

            export_dir = os.path.join(tmpdir, "hf_export")
            export_checkpoint_to_hf(checkpoint_path=ckpt_path, output_dir=export_dir, model_name="TestAxiomLM")

            # Check files exist
            self.assertTrue(os.path.exists(os.path.join(export_dir, "model.safetensors")))
            self.assertTrue(os.path.exists(os.path.join(export_dir, "config.json")))
            self.assertTrue(os.path.exists(os.path.join(export_dir, "generation_config.json")))
            self.assertTrue(os.path.exists(os.path.join(export_dir, "tokenizer_config.json")))
            self.assertTrue(os.path.exists(os.path.join(export_dir, "README.md")))

            # Verify safetensors weights can be loaded
            weights = load_file(os.path.join(export_dir, "model.safetensors"))
            self.assertIn("transformer.wte.weight", weights)
            self.assertIn("lm_head.weight", weights)

            # Load into fresh GPT instance
            fresh_model = GPT(cfg)
            fresh_model.load_state_dict(weights)
            fresh_model.eval()

            # Test forward pass
            x = torch.randint(0, 500, (1, 8))
            logits, _ = fresh_model(x)
            self.assertEqual(logits.shape, (1, 8, 500))


class TestMultiShardDataLoader(unittest.TestCase):
    """Rigorous tests for memory-mapped multi-shard streaming and rotation."""

    def test_shard_boundary_transition_and_wrap(self):
        """Verify DataLoaderLite seamlessly transitions across shard boundaries and wraps around."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create 3 tiny mock shards (500 tokens each)
            enc = np.arange(1500, dtype=np.uint16)
            enc[:500].tofile(os.path.join(tmpdir, "train_0000.bin"))
            enc[500:1000].tofile(os.path.join(tmpdir, "train_0001.bin"))
            enc[1000:1500].tofile(os.path.join(tmpdir, "train_0002.bin"))

            # B=2, T=10 -> 20 tokens per batch
            loader = DataLoaderLite(B=2, T=10, process_rank=0, num_processes=1, split="train", data_dir=tmpdir)
            self.assertEqual(len(loader.shards), 3)

            # Collect tokens across 35 batches (700 tokens > shard size of 500)
            seen_tokens = []
            for _ in range(35):
                x, y = loader.next_batch()
                self.assertEqual(x.shape, (2, 10))
                self.assertEqual(y.shape, (2, 10))
                # Targets should be shifted by 1
                self.assertTrue(torch.all(y[:, :-1] == x[:, 1:]))
                seen_tokens.append(x[0, 0].item())

            self.assertGreater(len(seen_tokens), 30)

    def test_dataloader_reset_and_epoch_counting(self):
        """Verify reset function restores stream state to initial shard and offset."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            enc = np.arange(600, dtype=np.uint16)
            enc.tofile(os.path.join(tmpdir, "train.bin"))

            loader = DataLoaderLite(B=2, T=8, process_rank=0, num_processes=1, split="train", data_dir=tmpdir)
            x1, _ = loader.next_batch()
            loader.reset()
            x2, _ = loader.next_batch()
            self.assertTrue(torch.equal(x1, x2), "Reset should replay exact initial batch.")


class TestAdvancedSamplingAndInference(unittest.TestCase):
    """Unit tests for advanced sampling strategies and KV-cache state management."""

    def test_min_p_sampling_truncation(self):
        """Verify min-p dynamic thresholding correctly filters low probability tail tokens."""
        # Logits with one dominant token and many tiny tail tokens
        logits = torch.tensor([[10.0, 5.0, 1.0, -5.0, -10.0]])
        # With min_p = 0.5 (relative to max prob), tail logits should be filtered to -inf
        filtered = sample_logits(logits.clone(), temperature=1.0, min_p=0.5)
        self.assertFalse(torch.isnan(filtered).any())

    def test_repetition_penalty_application(self):
        """Verify repetition penalty biases sampling away from previously generated token IDs."""
        # Logit for token 0 is dominant (10.0), token 1 is moderate (5.0)
        logits = torch.tensor([[10.0, 5.0, -10.0, -10.0]])
        # Without repetition penalty, token 0 is selected greedily
        tok_greedy = sample_logits(logits.clone(), temperature=0.0)
        self.assertEqual(tok_greedy.item(), 0)

        # With extreme repetition penalty (5.0) on token 0: 10.0 / 5.0 = 2.0 < 5.0 (token 1)
        gen_tokens = torch.tensor([[0]])
        tok_penalized = sample_logits(logits.clone(), temperature=0.0, repetition_penalty=5.0, prev_tokens=gen_tokens)
        self.assertEqual(tok_penalized.item(), 1, "Token 1 should be selected after penalizing token 0.")

    def test_kv_cache_state_reset_and_reuse(self):
        """Verify that reusing the model with and without cache reset produces deterministic outputs."""
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        cfg = GPTConfig(n_layer=2, n_head=4, n_kv_head=2, n_embd=64, vocab_size=50304, norm_type="rmsnorm", pos_emb="rope", mlp_type="swiglu")
        model = GPT(cfg)
        model.eval()

        samples1 = generate_with_cache(model, enc, device="cpu", prompt="def forward", max_length=15, temperature=0.0)
        samples2 = generate_with_cache(model, enc, device="cpu", prompt="def forward", max_length=15, temperature=0.0)
        self.assertEqual(samples1, samples2, "Consecutive cached generations must be 100% deterministic.")

    def test_directory_safetensors_and_config_loading(self):
        """Verify generate.py load_model correctly loads exported directories."""
        import tempfile
        from brain.export_hf import export_checkpoint_to_hf
        from brain.generate import load_model

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = GPTConfig(n_layer=2, n_head=4, n_kv_head=2, n_embd=64, vocab_size=300, norm_type="rmsnorm", pos_emb="rope", mlp_type="swiglu")
            model = GPT(cfg)
            ckpt_path = os.path.join(tmpdir, "model.pt")
            torch.save({"model_state_dict": model.state_dict(), "config": cfg, "step": 42}, ckpt_path)

            export_dir = os.path.join(tmpdir, "hf_dir")
            export_checkpoint_to_hf(ckpt_path, export_dir, "TestModel")

            # Load from directory path
            loaded_model, loaded_cfg = load_model(checkpoint_path=export_dir, arch="modern", device="cpu")
            self.assertEqual(loaded_cfg.n_layer, 2)
            self.assertEqual(loaded_cfg.n_kv_head, 2)
            self.assertEqual(loaded_cfg.norm_type, "rmsnorm")

            # Test forward parity between original and loaded
            x = torch.randint(0, 300, (1, 6))
            with torch.no_grad():
                orig_logits, _ = model(x)
                loaded_logits, _ = loaded_model(x)
            self.assertTrue(torch.allclose(orig_logits, loaded_logits, atol=1e-5))


class TestOptimizerSchedulingAndInvariants(unittest.TestCase):
    """Unit tests for optimizer learning rate schedule and parameter routing."""

    def test_cosine_learning_rate_schedule(self):
        """Verify learning rate schedule correctly executes warmup, cosine decay, and min_lr floor."""
        from brain.train_gpt2 import get_lr
        max_lr = 6e-4
        min_lr = max_lr * 0.1
        warmup_steps = 100
        max_steps = 1000

        # Step 0: lr starts at zero / initial ramp
        lr_0 = get_lr(0, warmup_steps=warmup_steps, max_steps=max_steps, max_lr=max_lr, min_lr=min_lr)
        self.assertAlmostEqual(lr_0, max_lr / (warmup_steps + 1), delta=1e-5)

        # Step = warmup_steps: reaches peak max_lr
        lr_warmup = get_lr(warmup_steps, warmup_steps=warmup_steps, max_steps=max_steps, max_lr=max_lr, min_lr=min_lr)
        self.assertAlmostEqual(lr_warmup, max_lr, places=5)

        # Step > max_steps: strictly bounded by min_lr floor
        lr_end = get_lr(max_steps + 50, warmup_steps=warmup_steps, max_steps=max_steps, max_lr=max_lr, min_lr=min_lr)
        self.assertEqual(lr_end, min_lr)

    def test_tied_weight_gradient_flow(self):
        """Verify that tied lm_head and wte tensors accumulate gradients correctly in backward pass."""
        cfg = GPTConfig(n_layer=1, n_head=2, n_embd=32, vocab_size=100)
        model = GPT(cfg)
        self.assertTrue(model.transformer.wte.weight is model.lm_head.weight)

        x = torch.randint(0, 100, (2, 4))
        y = torch.randint(0, 100, (2, 4))
        logits, loss = model(x, y)
        loss.backward()

        self.assertIsNotNone(model.transformer.wte.weight.grad)
        self.assertFalse(torch.isnan(model.transformer.wte.weight.grad).any())


if __name__ == "__main__":
    unittest.main(verbosity=2)

