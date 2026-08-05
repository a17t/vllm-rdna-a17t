# SPDX-License-Identifier: Apache-2.0
"""Correctness test for the RDNA2 (gfx1030) fp16 GEMV vs torch.linear.

Runs in seconds (no model load). Use this to confirm the Triton GEMV produces
correct output and does not trap on gfx1030 before restarting the full server.
"""

import pytest
import torch

from vllm.model_executor.layers.rdna2_gemm import rdna2_gemv
from vllm.platforms import current_platform
from vllm.platforms.rocm import on_gfx10x

DEVICE = torch.device("cuda")

# Shapes covering the unquantized decode projections (incl. Qwen3_5 GDN/attn).
SHAPES = [
    (5120, 5120),
    (13824, 5120),
    (5120, 13824),
    (4096, 4096),
    (1024, 4096),
    (256, 256),
]


@pytest.mark.skipif(
    not (current_platform.is_rocm() and on_gfx10x()),
    reason="Requires RDNA2 (gfx1030)",
)
@pytest.mark.parametrize("N,K", SHAPES)
@pytest.mark.parametrize("with_bias", [True, False])
def test_rdna2_gemv_vs_linear(N, K, with_bias):
    torch.manual_seed(0)
    x = torch.randn(K, dtype=torch.float16, device=DEVICE)
    w = torch.randn(N, K, dtype=torch.float16, device=DEVICE) * 0.1
    b = torch.randn(N, dtype=torch.float16, device=DEVICE) if with_bias else None

    ref = torch.nn.functional.linear(x, w, b)
    out = rdna2_gemv(x, w, b)

    assert out.shape == ref.shape
    assert out.isfinite().all(), "GEMV produced inf/nan (kernel may have trapped)"
    max_diff = (out.float() - ref.float()).abs().max().item()
    # fp16 GEMV vs fp32-accum reference: tolerant bound for large K.
    assert max_diff < 1.0, f"N={N} K={K} bias={with_bias}: max_diff={max_diff}"


@pytest.mark.skipif(
    not (current_platform.is_rocm() and on_gfx10x()),
    reason="Requires RDNA2 (gfx1030)",
)
def test_rdna2_gemv_smoke():
    """Smoke test: just confirm it runs without faulting on gfx1030."""
    x = torch.randn(128, dtype=torch.float16, device=DEVICE)
    w = torch.randn(64, 128, dtype=torch.float16, device=DEVICE)
    out = rdna2_gemv(x, w, None)
    assert out.shape == (64,)
    assert out.isfinite().all()
