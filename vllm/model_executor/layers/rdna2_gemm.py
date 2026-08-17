# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RDNA2 (gfx1030) fp16 GEMV for unquantized linear layers (decode path).

The generic skinny-GEMM kernels (wvSplitK/LLMM1) use gfx11/12 ISA that traps on
RDNA2, so unquantized decode projections otherwise fall through to
``torch.nn.functional.linear`` (~9% MBU on gfx1030). This module is a Triton
GEMV; Triton lowers it to gfx10-safe FMA (no WMMA/MFMA), the same codegen the
AWQ Triton path already relies on here.

Scope: the M=1 (single-row) case, which is the decode bottleneck, plus a
small-M (<= 16) tl.dot matmul for spec-decode verify / MTP batches
(ported from the gfx906 fork's skinny-GEMM tuning). Prefill and larger
batches keep using the existing path.
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


def _skinny_matmul_configs():
    # Ported from the gfx906 fork: shallow pipelining keeps LDS pressure low
    # on the 64 KB-LDS small dies; small N tiles expose more parallelism when
    # TP makes each shard narrow.
    cfgs = []
    for bn in (16, 32, 64, 128):
        for bk in (64, 128):
            for warps in (2, 4):
                cfgs.append(
                    triton.Config(
                        {"BLOCK_SIZE_N": bn, "BLOCK_SIZE_K": bk, "GROUP_SIZE_M": 1},
                        num_stages=1,
                        num_warps=warps,
                    )
                )
    return cfgs


@triton.autotune(
    configs=_skinny_matmul_configs(),
    key=["M", "N", "K"],
)
@triton.jit
def _skinny_matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """C = A x B.T with A (M, K), B (N, K), C (M, N)."""
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    c = accumulator.to(c_ptr.dtype.element_ty)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def rdna2_skinny_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Small-M matmul: out = a @ b.T with a (M, K), b (N, K).

    ``b`` is the nn.Linear weight in its native (N, K) layout; the kernel
    reads it with swapped strides so no repack is needed.
    """
    assert a.shape[1] == b.shape[1], "Incompatible dimensions"
    assert a.dtype == b.dtype, "Matrices A and B must have the same dtype"
    M, K = a.shape
    N = b.shape[0]
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(N, meta["BLOCK_SIZE_N"]),
    )
    _skinny_matmul_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(1),
        b.stride(0),
        c.stride(0),
        c.stride(1),
        BLOCK_SIZE_M=16,
        waves_per_eu=1,
    )
    return c


def _w4a16_gemv_configs():
    cfgs = []
    for bn in (64, 128, 256):
        for warps in (2, 4):
            cfgs.append(
                triton.Config({"BLOCK_N": bn}, num_warps=warps, num_stages=2)
            )
    return cfgs


@triton.autotune(
    configs=_w4a16_gemv_configs(),
    key=["N", "K", "GROUP"],
)
@triton.jit
def _w4a16_gemv_rdna2_kernel(
    x_ptr,  # [K] fp16/bf16 activations
    w_ptr,  # [K, N//8] int32, N-packed nibbles (triton_w4a16 layout)
    s_ptr,  # [K//GROUP, N] scales (contiguous)
    z_ptr,  # [K//GROUP, N//8] packed zeros if HAS_ZP
    b_ptr,  # [N] bias if HAS_BIAS
    y_ptr,  # [N] output
    N,
    K,
    N8,  # N // 8
    HAS_ZP: tl.constexpr,
    ZP_BIAS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    GROUP: tl.constexpr,  # quant group size (multiple of BLOCK_K)
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,  # == min group size
):
    """y = x @ dequant(W).T + bias for a single row (M=1 decode GEMV).

    Same packed layout as ``triton_w4a16_gemm``; unpacks each int32 into 8
    N-nibbles with the interleave+shift scheme and applies the per-group
    affine dequant inside the K loop. Measured ~171 GB/s effective on
    gfx1030 at gs=32 (the documented ~31% MBU ceiling for W4A16 GEMV on
    this part) — ~2.6x the generic tl.dot path at M=1.
    """
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_n8 = pid * (BLOCK_N // 8) + tl.arange(0, BLOCK_N // 8)
    n_mask = offs_n < N
    n8_mask = offs_n8 < N8

    shifts_row = tl.arange(0, 8) * 4
    shifts_2d = tl.broadcast_to(shifts_row[None, :], (BLOCK_N // 8, 8))
    shifts_1d = tl.reshape(shifts_2d, (BLOCK_N,))

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        ks = k0 + tl.arange(0, BLOCK_K)
        x = tl.load(x_ptr + ks).to(tl.float32)  # (BLOCK_K,)

        w_packed = tl.load(
            w_ptr + ks[:, None] * N8 + offs_n8[None, :],
            mask=n8_mask[None, :],
            other=0,
        )
        w = tl.interleave(w_packed, w_packed)
        w = tl.interleave(w, w)
        w = tl.interleave(w, w)
        w = (w >> shifts_1d[None, :]) & 0xF  # (BLOCK_K, BLOCK_N)

        g = k0 // GROUP
        s = tl.load(s_ptr + g * N + offs_n, mask=n_mask, other=0.0).to(
            tl.float32
        )
        if HAS_ZP:
            z_packed = tl.load(z_ptr + g * N8 + offs_n8, mask=n8_mask, other=0)
            z = tl.interleave(z_packed, z_packed)
            z = tl.interleave(z, z)
            z = tl.interleave(z, z)
            z = (z >> shifts_1d) & 0xF
            deq = (w.to(tl.float32) - z.to(tl.float32)[None, :]) * s[None, :]
        else:
            deq = (w.to(tl.float32) - ZP_BIAS) * s[None, :]
        acc += tl.sum(deq * x[:, None], axis=0)

    if HAS_BIAS:
        acc += tl.load(b_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + offs_n, acc.to(y_ptr.dtype.element_ty), mask=n_mask)


def rdna2_w4a16_gemv(
    x: torch.Tensor,
    w_q: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor | None,
    group_size: int,
    zp_bias: int = 8,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """M=1 W4A16 GEMV: y = x @ dequant(W).T (+ bias).

    ``w_q``/``scales``/``qzeros`` use the :func:`triton_w4a16_gemm` layouts
    ([K, N//8] int32, [K//G, N], [K//G, N//8]).
    """
    assert x.dim() == 1 or (x.dim() == 2 and x.shape[0] == 1)
    K = x.shape[-1]
    N = w_q.shape[1] * 8
    N8 = N // 8
    assert w_q.shape == (K, N8)
    assert scales.shape == (K // group_size, N)
    if qzeros is not None:
        assert qzeros.shape == (K // group_size, N8)

    x1 = x.reshape(K)
    y = torch.empty(N, dtype=x.dtype, device=x.device)
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]),)
    _w4a16_gemv_rdna2_kernel[grid](
        x1,
        w_q,
        scales,
        qzeros if qzeros is not None else w_q,
        bias if bias is not None else x1,
        y,
        N,
        K,
        N8,
        HAS_ZP=qzeros is not None,
        ZP_BIAS=zp_bias,
        HAS_BIAS=bias is not None,
        GROUP=group_size,
        BLOCK_K=min(32, group_size),
    )
    return y
