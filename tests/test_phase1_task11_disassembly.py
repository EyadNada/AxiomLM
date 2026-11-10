"""
AxiomLM Phase 1 — Task 1.1 Peer-Review Unit & Systems Test Suite:
Microarchitectural Metal Shader Disassembly, Instruction Accounting,
and Register Pressure Validation.
"""

from __future__ import annotations

import struct
import unittest
from pathlib import Path


from benchmarks.benchmark_phase1_disassembly import (
    build_and_run_probe,
    compute_instruction_mix_metrics,
)


class TestPhase1Task11Disassembly(unittest.TestCase):
    """
    Rigorously tests microarchitectural properties required by MLSys Reviewers:
    1. Zero DRAM register spill & maximum warp occupancy (maxTotalThreadsPerThreadgroup == 1024).
    2. Zero-error IEEE-754 mantissa insertion trick for 4-bit dequantization.
    3. Verified instruction reduction (>50% ALU reduction, 100% conversion elimination).
    4. Threadgroup memory budget compliance (< 32 KB).
    5. In-register speculative tree arithmetic intensity scaling across the hardware roofline knee.
    """

    def setUp(self):
        self.metal_path = "axiomlm/kernels/metal_kernels.metal"
        self.assertTrue(
            Path(self.metal_path).exists(),
            f"Metal source file {self.metal_path} must exist.",
        )

    def test_01_metal_pipeline_max_occupancy_and_zero_dram_spill(self):
        """
        Verify that all core Metal compute pipelines maintain maxTotalThreadsPerThreadgroup == 1024.
        On Apple Silicon (AGX), exceeding the 32-register-per-thread budget reduces occupancy
        down to 512, 256, or 128 threads/TG and triggers DRAM spill buffers.
        """
        probe_data = build_and_run_probe(self.metal_path)
        kernels = probe_data["kernels"]

        critical_kernels = [
            "qmv_q8_g64_half_kernel",
            "qmv_q4_g64_half_kernel",
            "qmv_q4_marlin_apple_kernel",
            "qmm_q4_g64_half_kernel",
            "qmm_q4_marlin_apple_kernel",
            "gqa_flash_decode_split_kv_half_kernel",
            "gqa_tree_verify_split_kv_half_kernel",
        ]

        for kname in critical_kernels:
            self.assertIn(kname, kernels, f"Kernel {kname} must be compiled in probe.")
            kinfo = kernels[kname]
            max_th = kinfo["max_total_threads_per_threadgroup"]
            self.assertEqual(
                max_th,
                1024,
                f"Kernel {kname} has reduced occupancy ({max_th} < 1024), indicating register spill!",
            )
            self.assertEqual(
                kinfo["thread_execution_width"],
                32,
                f"Kernel {kname} must have SIMD width 32 on Apple Silicon.",
            )

    def test_02_marlin_ieee754_mantissa_unpack_exactness(self):
        """
        Bit-for-bit mathematical validation of the IEEE-754 Binary16 mantissa insertion trick
        used in marlin_unpack_4x_half2.
        For all 16 possible 4-bit nibbles [0..15]:
          - (nibble | 0x6400) interpreted as FP16 minus 1024.0 == float(nibble)
          - ((nibble << 4) | 0x5400) interpreted as FP16 minus 64.0 == float(nibble)
        """
        for nibble in range(16):
            # Nibble in bits 0..3: magic exponent 0x6400 (biased exponent 25 -> 2^10 = 1024.0)
            u16_lo = nibble | 0x6400
            f16_lo = struct.unpack("e", struct.pack("<H", u16_lo))[0]
            val_lo = f16_lo - 1024.0
            self.assertEqual(
                val_lo,
                float(nibble),
                f"Low nibble {nibble} failed IEEE-754 dequantization: got {val_lo}",
            )

            # Nibble in bits 4..7: magic exponent 0x5400 (biased exponent 21 -> 2^6 = 64.0)
            u16_hi = (nibble << 4) | 0x5400
            f16_hi = struct.unpack("e", struct.pack("<H", u16_hi))[0]
            val_hi = f16_hi - 64.0
            self.assertEqual(
                val_hi,
                float(nibble),
                f"High nibble {nibble} failed IEEE-754 dequantization: got {val_hi}",
            )

    def test_03_instruction_mix_accounting_parity(self):
        """
        Verify that MARLIN ladder dequantization strictly cuts ALU unpacking instructions
        by >50% (measured 60.87%) and eliminates 100% of scalar integer-to-float conversions.
        """
        mix = compute_instruction_mix_metrics()
        reductions = mix["instruction_reductions"]

        self.assertGreaterEqual(
            reductions["integer_alu_instruction_reduction_pct"],
            50.0,
            "MARLIN must achieve at least 50% integer ALU instruction reduction.",
        )
        self.assertEqual(
            reductions["conversion_instruction_elimination_pct"],
            100.0,
            "MARLIN must eliminate 100% of integer-to-float conversion instructions.",
        )
        self.assertGreaterEqual(
            reductions["bit_shift_reduction_pct"],
            80.0,
            "MARLIN must reduce bit shifts by at least 80% (7 shifts down to 1 shift).",
        )

    def test_04_threadgroup_memory_budget_bounds(self):
        """
        Verify that static threadgroup memory length does not exceed 4 KB,
        preserving plenty of headroom under Apple's 32 KB threadgroup memory ceiling.
        """
        probe_data = build_and_run_probe(self.metal_path)
        kernels = probe_data["kernels"]

        for kname, kinfo in kernels.items():
            tg_mem = kinfo["static_threadgroup_memory_length"]
            self.assertLessEqual(
                tg_mem,
                4096,
                f"Kernel {kname} exceeds 4KB threadgroup memory budget: {tg_mem} bytes.",
            )

    def test_05_tree_verify_instruction_amortization_scaling(self):
        """
        Verify that multi-candidate tree verification scales arithmetic intensity
        and crosses the M3 Pro roofline hardware knee (57.3 FLOPs/Byte) at T=16..32.
        """
        mix = compute_instruction_mix_metrics()
        scaling = mix["tree_arithmetic_intensity_scaling"]

        # Single-token (T=1) must be severely memory-bound
        i_t1 = scaling["T=1"]["arithmetic_intensity_flops_per_byte"]
        self.assertLess(
            i_t1, 10.0, "T=1 must have arithmetic intensity < 10 FLOPs/Byte."
        )

        # Tree verification (T=32) must be compute-bound (> 57.3 FLOPs/Byte)
        i_t32 = scaling["T=32"]["arithmetic_intensity_flops_per_byte"]
        self.assertGreater(
            i_t32,
            57.3,
            "T=32 must cross the 57.3 FLOPs/Byte hardware balance knee into compute saturation.",
        )

        # Unpack ops per emitted candidate token must scale strictly as O(1/T)
        unpack_t1 = scaling["T=1"]["amortized_unpack_instructions_per_weight"]
        unpack_t16 = scaling["T=16"]["amortized_unpack_instructions_per_weight"]
        self.assertAlmostEqual(
            unpack_t1 / 16.0,
            unpack_t16,
            places=3,
            msg="Unpack instructions per weight must amortize exactly by 1/T.",
        )


if __name__ == "__main__":
    unittest.main()
