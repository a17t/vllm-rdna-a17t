# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RDNA2 (gfx1030) fp16 GEMV for unquantized linear layers (decode path).

The generic skinny-GEMM kernels (wvSplitK/LLMM1) use gfx11/12 ISA that traps on
RDNA2, so unquantized decode projections otherwise fall through to
``torch.nn.functional.linear`` (~9% MBU on gfx1030). This module is a Triton
GEMV; Triton lowers it to gfx10-safe FMA (no WMMA/MFMA), the same codegen the
AWQ Triton path already relies on here.

Scope: the M=1 (single-row) case, which is the decode bottleneck. Prefill and
small-batch (M>1) keep using the existing path.
"""

import torch

from vllm.triton_utils import tl, triton


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_N": 256, "BLOCK_K": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_N": 64, "BLOCK_K": 128}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_N": 256, "BLOCK_K": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 256}, num_warps=8, num_stages=3),
    ],
    key=["N", "K"],
)
@triton.jit
def _gemv_rdna2_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    N,
    K,
    stride_wn,
    stride_wk,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    pid_n = tl.program_id(0)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)
        k_mask = k < K
        x = tl.load(x_ptr + k, mask=k_mask, other=0.0).to(tl.float32)
        w = tl.load(
            w_ptr + offs_n[:, None] * stride_wn + k[None, :] * stride_wk,
            mask=n_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(w * x[None, :], axis=1)

    if HAS_BIAS:
        b = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
        acc += b

    tl.store(out_ptr + offs_n, acc.to(out_ptr.dtype.element_ty), mask=n_mask)


def rdna2_gemv(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None
) -> torch.Tensor:
    """fp16/bf16 GEMV for M=1: out = x @ weight.T (+ bias).

    Args:
        x: 1D activation [K].
        weight: 2D [N, K] (nn.Linear layout, ``out = x @ weight.T``).
        bias: optional [N].
    """
    N, K = weight.shape
    out = torch.empty(N, dtype=x.dtype, device=x.device)
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]),)
    _gemv_rdna2_kernel[grid](
        x,
        weight,
        bias,
        out,
        N,
        K,
        weight.stride(0),
        weight.stride(1),
        HAS_BIAS=bias is not None,
    )
    return out
