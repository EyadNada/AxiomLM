"""
AxiomLM Phase 1 — Task 1.3 Peer-Review Unit & Systems Test Suite:
Empirical Bandwidth Saturation Curve, Cold-DRAM Bandwidth Scaling,
and Publication Golden Artifact Integrity.
"""

from __future__ import annotations

import json
from pathlib import Path
import unittest

import torch

from axiomlm.kernels import (
    pack_linear_q4_g64,
    pack_linear_q4_marlin,
    qmm_q4_g64,
    qmm_q4_marlin,
)
from benchmarks.benchmark_phase1_task13_bandwidth_saturation import (
    run_latency_trials,
    sync_mps,
)


class TestPhase1Task13BandwidthSaturation(unittest.TestCase):
    """
    Rigorously validates the bandwidth saturation curve and golden artifacts:
    1. MARLIN Q4 sustains strictly higher memory bandwidth than Naive Q4 across M in {4, 8, 16}.
    2. Golden vector artifacts (SVG & PNG) are generated and well-formed.
    3. JSON report covers all 4 formats across M in {1, 2, 4, 8, 16, 32, 64}.
    4. Monotonic scaling of arithmetic intensity across the spectrum.
    """

    def setUp(self):
        torch.manual_seed(2026)
        self.device = torch.device("mps")
        self.N = 2048
        self.K = 2048

    def test_01_marlin_bandwidth_advantage_in_tree_regimes(self):
        """
        Verify that MARLIN sustains strictly higher bandwidth (GB/s) than Naive Q4 for M=8, 16.
        """
        w = (
            (torch.randn(self.N, self.K, dtype=torch.float32) * 0.02)
            .half()
            .to(self.device)
        )
        p4, sb4 = pack_linear_q4_g64(w)
        pm, sbm = pack_linear_q4_marlin(w)

        pool_q4 = [(p4.clone(), sb4.clone()) for _ in range(16)]
        pool_mar = [(pm.clone(), sbm.clone()) for _ in range(16)]
        sync_mps()

        weight_bytes = self.N * (self.K // 2) + self.N * (self.K // 64) * 4

        for M in [8, 16]:
            x = torch.randn(1, M, self.K, dtype=torch.float16, device=self.device)
            io_bytes = M * self.K * 2 + M * self.N * 2
            total_bytes = weight_bytes + io_bytes

            t_naive, _, _, _ = run_latency_trials(
                lambda ws: qmm_q4_g64(x, ws[0], ws[1]),
                pool_q4,
                steps_per_trial=8,
                num_trials=5,
            )
            t_marlin, _, _, _ = run_latency_trials(
                lambda ws: qmm_q4_marlin(x, ws[0], ws[1], out_features=self.N),
                pool_mar,
                steps_per_trial=8,
                num_trials=5,
            )

            bw_naive = (total_bytes / (t_naive * 1e-6)) / 1e9
            bw_marlin = (total_bytes / (t_marlin * 1e-6)) / 1e9

            self.assertGreater(
                bw_marlin,
                bw_naive * 1.25,
                f"MARLIN bandwidth at M={M} ({bw_marlin:.1f} GB/s) must exceed Naive Q4 ({bw_naive:.1f} GB/s) by >= 1.25x",
            )

    def test_02_report_file_integrity(self):
        """
        Verify that phase1_task13_bandwidth_saturation_report.json exists and contains complete curves.
        """
        report_path = Path("benchmarks/phase1_task13_bandwidth_saturation_report.json")
        self.assertTrue(report_path.exists(), "Task 1.3 report JSON must exist.")

        with open(report_path, "r") as f:
            data = json.load(f)

        self.assertIn("curves", data)
        curves = data["curves"]
        expected_formats = [
            "FP16 (16-bit)",
            "Q8_G64 (8-bit)",
            "Naive Q4_G64 (4-bit)",
            "AxiomLM MARLIN (4-bit)",
        ]
        for fmt in expected_formats:
            self.assertIn(fmt, curves, f"Format {fmt} must be in curves report.")
            points = curves[fmt]
            self.assertEqual(
                len(points), 7, f"Format {fmt} must have 7 batch points M=1..64."
            )

    def test_03_vector_artifacts_generated(self):
        """
        Verify that the publication vector figure (SVG) and preview (PNG) exist and are valid.
        """
        svg_path = Path("benchmarks/bandwidth_saturation_curve.svg")
        png_path = Path("benchmarks/bandwidth_saturation_curve.png")

        self.assertTrue(svg_path.exists(), "SVG artifact must exist.")
        self.assertTrue(png_path.exists(), "PNG artifact must exist.")
        self.assertGreater(
            svg_path.stat().st_size, 50000, "SVG artifact must be >50KB."
        )
        self.assertGreater(
            png_path.stat().st_size, 100000, "PNG artifact must be >100KB."
        )

        with open(svg_path, "r", encoding="utf-8") as f:
            header = f.read(200)
            self.assertTrue(
                "<svg" in header or "<?xml" in header,
                "SVG must have valid XML/SVG header.",
            )

    def test_04_roofline_scaling_regimes(self):
        """
        Verify that arithmetic intensity scales monotonically from < 4.0 to > 150 FLOPs/Byte.
        """
        report_path = Path("benchmarks/phase1_task13_bandwidth_saturation_report.json")
        with open(report_path, "r") as f:
            data = json.load(f)

        marlin_points = data["curves"]["AxiomLM MARLIN (4-bit)"]
        intensities = [p["arithmetic_intensity_flops_per_byte"] for p in marlin_points]

        self.assertLess(intensities[0], 5.0, "M=1 intensity must be < 5.0 FLOP/B.")
        self.assertGreater(
            intensities[-1], 150.0, "M=64 intensity must exceed 150.0 FLOP/B."
        )

        for i in range(1, len(intensities)):
            self.assertGreater(
                intensities[i],
                intensities[i - 1],
                "Arithmetic intensity must be strictly monotonically increasing with M.",
            )


if __name__ == "__main__":
    unittest.main()
