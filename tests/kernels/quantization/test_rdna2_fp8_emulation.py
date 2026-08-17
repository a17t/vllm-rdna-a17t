# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for the gfx10x FP8 emulation paths (no native FP8).

Covers the W8A16-emulated block-scaled GEMM (fp16 activations + bitwise
E4M3 weight decode) and the fp16-activation fused MoE kernel.
"""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.platforms.rocm import on_gfx10x

if not (current_platform.is_rocm() and on_gfx10x()):
    pytest.skip("Requires RDNA2 (gfx1030)", allow_module_level=True)

DEVICE = torch.device("cuda")


def _decode_e4m3_fp16(b_uint8: torch.Tensor) -> torch.Tensor:
    """Reference bitwise E4M3 -> fp16 decode (subnormals -> 0)."""
    b = b_uint8.cpu().to(torch.int32)
    sign = (b & 0x80) << 8
    exp = (b & 0x78) >> 3
    exp = torch.where(exp == 0, torch.zeros_like(exp), exp + 8)
    mant = (b & 0x07) << 7
    bits = sign | (exp << 10) | mant
    return bits.to(torch.uint16).view(torch.float16).to(b_uint8.device)


def _dequant_block(b_fp8: torch.Tensor, bs: torch.Tensor, bn: int, bk: int):
    """Expand block scales and decode to fp16 weights (2-D or 3-D)."""
    b_h = _decode_e4m3_fp16(b_fp8.view(torch.uint8)).float()
    scale = bs.float()
    n_last = b_h.shape[-2] // bs.shape[-2]
    k_last = b_h.shape[-1] // bs.shape[-1]
    scale = scale.repeat_interleave(n_last, dim=-2).repeat_interleave(k_last, dim=-1)
    return (b_h * scale).to(torch.float16)


@pytest.mark.parametrize("M", [1, 3, 17, 128])
@pytest.mark.parametrize("N,K", [(512, 1024), (2048, 4096)])
def test_w8a8_block_fp8_fp16_act(M, N, K):
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        w8a8_triton_block_scaled_mm,
    )

    torch.manual_seed(0)
    bn, bk = 128, 128
    a = torch.randn(M, K, dtype=torch.float16, device=DEVICE)
    b_fp8 = (
        (torch.randn(N, K, device=DEVICE) * 0.05)
        .clamp(-448, 448)
        .to(torch.float8_e4m3fn)
    )
    bs = torch.rand(N // bn, K // bk, dtype=torch.float32, device=DEVICE) * 0.01 + 0.005
    as_placeholder = a.new_empty(1)

    out = w8a8_triton_block_scaled_mm(
        a, b_fp8, as_placeholder, bs, [bn, bk], torch.float16
    )
    assert out.isfinite().all(), "FP8-emulated GEMM produced inf/nan"

    ref = (a.float() @ _dequant_block(b_fp8, bs, bn, bk).float().T).to(torch.float16)
    torch.testing.assert_close(out, ref, atol=5e-2, rtol=5e-2)


def test_fused_moe_fp8_fp16_act():
    from vllm.model_executor.layers.fused_moe.fused_moe import (
        invoke_fused_moe_triton_kernel,
    )
    from vllm.triton_utils import tl

    torch.manual_seed(0)
    E, N, K, top_k, M = 8, 256, 512, 2, 32
    bn, bk = 128, 128
    a = torch.randn(M, K, dtype=torch.float16, device=DEVICE)
    w = (
        (torch.randn(E, N, K, device=DEVICE) * 0.05)
        .clamp(-448, 448)
        .to(torch.float8_e4m3fn)
    )
    ws = (
        torch.rand(E, N // bn, K // bk, dtype=torch.float32, device=DEVICE) * 0.01
        + 0.005
    )
    topk_weights = torch.rand(M, top_k, dtype=torch.float32, device=DEVICE)
    topk_weights = topk_weights / topk_weights.sum(-1, keepdim=True)
    topk_ids = torch.randint(0, E, (M, top_k), dtype=torch.int32, device=DEVICE)

    c = torch.zeros(M, top_k, N, dtype=torch.float16, device=DEVICE)
    sorted_ids = torch.arange(M * top_k, dtype=torch.int32, device=DEVICE)
    expert_ids = topk_ids[:, 0].contiguous()
    num_tokens_padded = torch.tensor([M * top_k], dtype=torch.int32, device=DEVICE)
    config = {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 1,
        "num_warps": 4,
        "num_stages": 1,
    }

    invoke_fused_moe_triton_kernel(
        a,
        w,
        c,
        None,
        ws,
        topk_weights,
        sorted_ids,
        expert_ids,
        num_tokens_padded,
        True,
        top_k,
        config,
        tl.float16,
        use_fp8_w8a8=True,
        use_int8_w8a8=False,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        per_channel_quant=False,
        block_shape=[bn, bk],
    )
    assert c.isfinite().all(), "FP8-emulated fused MoE produced inf/nan"

    w_ref = _dequant_block(w, ws, bn, bk).float()
    for i in range(M):
        for j in range(top_k):
            e = int(topk_ids[i, j])
            ref = (a[i].float() @ w_ref[e].T) * float(topk_weights[i, j])
            torch.testing.assert_close(
                c[i, j], ref.to(torch.float16), atol=5e-2, rtol=5e-2
            )
