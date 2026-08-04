# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Correctness test for RDNA2 AWQ native kernel vs Triton ground truth.

Tests the native HIP kernel (awq_gemm_rdna2) against the Triton kernel
(triton_w4a16_gemm) using identical quantized weight layouts.
"""

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.model_executor.kernels.linear.mixed_precision.triton_w4a16 import (
    triton_w4a16_gemm,
)
from vllm.platforms import current_platform
from vllm.platforms.rocm import on_gfx10x


@pytest.fixture(scope="module")
def device():
    return torch.device("cuda")


@pytest.fixture(scope="module")
def has_rdna2():
    return current_platform.is_rocm() and on_gfx10x()


@pytest.fixture(scope="module")
def has_awq_op():
    return hasattr(torch.ops, "_rocm_C") and hasattr(
        torch.ops._rocm_C, "awq_gemm_rdna2"
    )


TEST_CASES = [
    # (M, K, N, group_size)
    (1, 1152, 1152, 128),  # Qwen3.5-0.8B decode
    (1, 4096, 4096, 128),  # typical decode
    (1, 1152, 3072, 128),  # qkv projection
    (8, 1152, 1152, 128),  # small prefill
    (32, 1152, 1152, 128),  # medium prefill
    (128, 1152, 1152, 128),  # large prefill
    (1, 4096, 1024, 64),  # different group size
    (1, 4096, 4096, 256),  # different group size
    (1, 1024, 2048, 128),  # rectangular
    (1, 2048, 1024, 128),  # rectangular inverse
    (1, 256, 256, 64),  # K div by groupsize, tail loop exercises K=248..255
    (1, 128, 256, 128),  # K == groupsize, single group, no tail
    (1, 256, 256, 128),  # K = 2*groupsize, 2 groups, no tail
    # Qwen3-27B-class decode shapes (M=1).
    (1, 5120, 5120, 128),  # hidden-size projection
    (1, 5120, 13824, 128),  # gate/up projection
    (1, 13824, 5120, 128),  # down projection
    (1, 5120, 1024, 128),  # N not a multiple of BLOCK_N*4 (tail block)
]


def _make_random_quantized(K, N, group_size, device):
    """Create random quantized weights in [K, N//8] layout (sequential)."""
    groups = K // group_size
    N8 = N // 8

    # Random int4 weights [K, N//8] sequential packing
    w_unpacked = torch.randint(0, 16, (K, N), dtype=torch.int32, device=device)
    shifts = torch.arange(8, device=device, dtype=torch.int32) * 4
    w_packed = torch.sum(
        (w_unpacked.view(K, N8, 8) & 0xF) << shifts,
        dim=2,
        dtype=torch.int32,
    )

    # Random zero points [G, N//8] sequential packing
    z_unpacked = torch.randint(0, 16, (groups, N), dtype=torch.int32, device=device)
    z_packed = torch.sum(
        (z_unpacked.view(groups, N8, 8) & 0xF) << shifts,
        dim=2,
        dtype=torch.int32,
    )

    # Random scales [G, N] fp16 — use small positive values
    scales = torch.rand((groups, N), dtype=torch.float16, device=device) * 0.01 + 1e-4

    return w_packed, z_packed, scales


@pytest.mark.skipif(not on_gfx10x(), reason="Requires RDNA2 (gfx1030)")
@pytest.mark.skipif(
    not (
        hasattr(torch.ops, "_rocm_C") and hasattr(torch.ops._rocm_C, "awq_gemm_rdna2")
    ),
    reason="awq_gemm_rdna2 op not available",
)
class TestRDNA2AWQCorrectness:
    """Compare native AWQ kernel output against Triton ground truth."""

    @pytest.mark.parametrize("M,K,N,group_size", TEST_CASES)
    def test_awq_vs_triton(self, M, K, N, group_size, device):
        """Native kernel output must match Triton within fp16 tolerance."""
        w_packed, z_packed, scales = _make_random_quantized(K, N, group_size, device)

        # Random fp16 activations
        a = torch.randn((M, K), dtype=torch.float16, device=device)

        # Triton ground truth
        triton_out = triton_w4a16_gemm(
            a=a,
            b_q=w_packed,
            scales=scales,
            qzeros=z_packed,
            group_size=group_size,
            zp_bias=0,
        )

        # Native kernel
        native_out = ops.awq_gemm_rdna2(a, w_packed, z_packed, scales)

        # Compare
        assert native_out.shape == triton_out.shape, (
            f"Shape mismatch: {native_out.shape} vs {triton_out.shape}"
        )

        max_diff = (native_out.float() - triton_out.float()).abs().max().item()
        mean_diff = (native_out.float() - triton_out.float()).abs().mean().item()

        # fp16 tolerance: max per-element error < 0.05
        assert max_diff < 0.05, (
            f"M={M} K={K} N={N} G={group_size}: "
            f"max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f} "
            f"(threshold=0.05)"
        )

    @pytest.mark.parametrize(
        "M,K,N,group_size",
        [
            (1, 1152, 1152, 128),
            (1, 4096, 4096, 128),
            (32, 4096, 4096, 128),
        ],
    )
    def test_awq_numerical_sanity(self, M, K, N, group_size, device):
        """Verify output is not all-zeros and has reasonable magnitude."""
        w_packed, z_packed, scales = _make_random_quantized(K, N, group_size, device)
        a = torch.randn((M, K), dtype=torch.float16, device=device)

        out = ops.awq_gemm_rdna2(a, w_packed, z_packed, scales)

        assert not (out == 0).all(), "Output is all zeros"
        assert out.isfinite().all(), "Output contains inf/nan"
        assert out.abs().max().item() < 1e4, (
            f"Output magnitude too large: {out.abs().max().item()}"
        )

    @pytest.mark.parametrize(
        "K,N,group_size",
        [
            (1152, 1152, 128),
            (4096, 4096, 128),
            (5120, 13824, 128),
            (13824, 5120, 128),
            (5120, 1024, 128),  # tail block (N not a multiple of 256)
            (256, 256, 64),  # group boundary inside a K tile
        ],
    )
    def test_gemv_vs_batched_parity(self, K, N, group_size, device):
        """M=1 GEMV path must match the M=2 batched kernel for the same row.

        The M=1 call routes to the dedicated GEMV kernel; the M=2 call uses
        the original atomic-CAS batched kernel. Row 0 of the M=2 output must
        match the M=1 output within a tight tolerance (same dequant math,
        only reduction order differs).
        """
        w_packed, z_packed, scales = _make_random_quantized(K, N, group_size, device)
        a = torch.randn((1, K), dtype=torch.float16, device=device)
        # Pad to M=2 for the batched path; row 1 is irrelevant.
        a2 = torch.cat([a, torch.zeros(1, K, dtype=torch.float16, device=device)])

        out_gemv = ops.awq_gemm_rdna2(a, w_packed, z_packed, scales)
        out_batched = ops.awq_gemm_rdna2(a2, w_packed, z_packed, scales)[0:1]

        max_diff = (out_gemv.float() - out_batched.float()).abs().max().item()
        # Same arithmetic, only loop ordering differs -> very tight bound.
        assert max_diff < 1e-3, (
            f"K={K} N={N} G={group_size}: GEMV vs batched max_diff={max_diff:.6f}"
        )
