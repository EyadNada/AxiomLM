"""
Quantization SNR, Cosine Similarity, and Numerical Drift Tests.
Rigorously evaluates error bounds and statistical distribution fidelity
for Q8_G64 and Q4_G64 quantization against unquantized FP16 baselines.
"""

import math
import unittest
import torch
import torch.nn.functional as F

from axiomlm.models.transformer import Transformer, ModelConfig
from axiomlm.kernels import (
    pack_linear_q4_g64,
    dequantize_q4_g64,
    pack_linear_q8_g64,
    dequantize_q8_g64,
)


class TestQuantizationDriftAndFidelity(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1337)
        self.device = (
            torch.device("mps")
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            else torch.device("cpu")
        )

    def _create_model(self) -> Transformer:
        config = ModelConfig(
            arch="modern",
            n_layer=4,
            n_head=4,
            n_kv_head=2,
            n_embd=256,
            vocab_size=2000,
            block_size=256,
            norm_type="rmsnorm",
            mlp_type="swiglu",
            pos_emb="rope",
            use_fused_kernels=(self.device.type == "mps"),
        )
        return Transformer(config).to(device=self.device, dtype=torch.float16).eval()

    def test_per_layer_quantization_snr_and_cosine_similarity(self):
        """
        Calculates per-tensor SNR (dB) and Cosine Similarity for every projection
        matrix across Q4_G64 and Q8_G64 formats.
        """
        shapes = [
            ("attn.c_attn", (448, 256)),  # QKV: 256 + 2*96
            ("attn.c_proj", (256, 256)),
            ("mlp.gate_up", (1408, 256)),
            ("mlp.w_down", (256, 704)),
            ("lm_head", (2000, 256)),
        ]

        for name, shape in shapes:
            W = torch.randn(shape, device=self.device, dtype=torch.float16)

            # --- Q8_G64 ---
            packed_q8, sb_q8 = pack_linear_q8_g64(W)
            w_hat_q8 = dequantize_q8_g64(packed_q8, sb_q8, shape[1])

            cos_sim_q8 = F.cosine_similarity(
                W.view(-1), w_hat_q8.view(-1), dim=0
            ).item()
            noise_q8 = (W - w_hat_q8).pow(2).sum().item()
            signal = W.pow(2).sum().item()
            snr_db_q8 = 10.0 * math.log10(signal / (noise_q8 + 1e-12))

            self.assertGreater(
                cos_sim_q8,
                0.999,
                f"Q8 Cosine similarity for {name} ({cos_sim_q8:.5f}) below 0.999 threshold",
            )
            self.assertGreater(
                snr_db_q8,
                35.0,
                f"Q8 SNR for {name} ({snr_db_q8:.2f} dB) below 35.0 dB threshold",
            )

            # --- Q4_G64 ---
            packed_q4, sb_q4 = pack_linear_q4_g64(W)
            w_hat_q4 = dequantize_q4_g64(packed_q4, sb_q4, shape[1])

            cos_sim_q4 = F.cosine_similarity(
                W.view(-1), w_hat_q4.view(-1), dim=0
            ).item()
            noise_q4 = (W - w_hat_q4).pow(2).sum().item()
            snr_db_q4 = 10.0 * math.log10(signal / (noise_q4 + 1e-12))

            self.assertGreater(
                cos_sim_q4,
                0.990,
                f"Q4 Cosine similarity for {name} ({cos_sim_q4:.5f}) below 0.990 threshold",
            )
            self.assertGreater(
                snr_db_q4,
                20.0,
                f"Q4 SNR for {name} ({snr_db_q4:.2f} dB) below 20.0 dB threshold",
            )

    def test_output_distribution_kl_divergence(self):
        """
        Measures statistical divergence between full FP16 predictions and
        quantized (Q8 and Q4) logits over realistic context tokens.
        """
        # Create models with identical seeds
        torch.manual_seed(42)
        model_fp16 = self._create_model()
        prompt = torch.randint(0, 2000, (1, 64), device=self.device, dtype=torch.long)

        with torch.no_grad():
            logits_fp16, _ = model_fp16(prompt)
            prob_fp16 = F.softmax(logits_fp16[:, -1, :], dim=-1)

        # Clone and quantize to Q8
        torch.manual_seed(42)
        model_q8 = self._create_model()
        model_q8.quantize("q8")

        with torch.no_grad():
            logits_q8, _ = model_q8(prompt)
            prob_q8 = F.softmax(logits_q8[:, -1, :], dim=-1)

        # Clone and quantize to Q4
        torch.manual_seed(42)
        model_q4 = self._create_model()
        model_q4.quantize("q4")

        with torch.no_grad():
            logits_q4, _ = model_q4(prompt)
            prob_q4 = F.softmax(logits_q4[:, -1, :], dim=-1)

        # KL Divergence D_KL(P || Q) = sum(P * log(P / Q))
        kl_q8 = F.kl_div(
            prob_q8.log(), prob_fp16, reduction="batchmean", log_target=False
        ).item()
        kl_q4 = F.kl_div(
            prob_q4.log(), prob_fp16, reduction="batchmean", log_target=False
        ).item()

        self.assertLess(
            kl_q8,
            0.05,
            f"Q8 KL Divergence ({kl_q8:.4f} nats) exceeded 0.05 nats threshold",
        )
        self.assertLess(
            kl_q4,
            0.35,
            f"Q4 KL Divergence ({kl_q4:.4f} nats) exceeded 0.35 nats threshold",
        )

    def test_long_horizon_stability_no_nan_or_inf(self):
        """
        Generates 64 consecutive tokens autoregressively under Q4 quantization.
        Verifies that accumulated quantized activations never produce NaNs or Infs.
        """
        model = self._create_model()
        model.quantize("q4")

        prompt = torch.randint(0, 2000, (1, 16), device=self.device, dtype=torch.long)
        curr = prompt

        with torch.no_grad():
            for step in range(64):
                logits, _ = model(curr)
                last_logit = logits[:, -1, :]
                self.assertFalse(
                    torch.isnan(last_logit).any(), f"NaN detected at step {step}"
                )
                self.assertFalse(
                    torch.isinf(last_logit).any(), f"Inf detected at step {step}"
                )
                next_tok = last_logit.argmax(dim=-1, keepdim=True)
                curr = torch.cat([curr, next_tok], dim=1)


if __name__ == "__main__":
    unittest.main()
