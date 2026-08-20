# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheme-path check: compressed-tensors W4 layer through the RDNA2
Triton W4A16 kernel (both packing layouts) vs dequant reference."""

import pytest
import torch

from vllm.model_executor.kernels.linear.mixed_precision.MPLinearKernel import (
    MPLinearLayerConfig,
)
from vllm.model_executor.kernels.linear.mixed_precision.rdna2_triton_w4a16 import (
    RDNA2TritonW4A16LinearKernel,
)
from vllm.model_executor.parameter import (
    GroupQuantScaleParameter,
    PackedvLLMParameter,
)
from vllm.platforms import current_platform
from vllm.platforms.rocm import on_gfx10x
from vllm.scalar_type import scalar_types


class _FakeLayer(torch.nn.Module):
    pass


@pytest.mark.skipif(
    not (current_platform.is_rocm() and on_gfx10x()),
    reason="Requires RDNA2 (gfx1030)",
)
@pytest.mark.parametrize("use_kpack", [True, False])
def test_w4a16_scheme_path_roundtrip(default_vllm_config, use_kpack, monkeypatch):
    N, K, GS = 512, 1024, 32
    torch.manual_seed(0)
    import torch.distributed as dist
    if not dist.is_initialized():
        from tests.utils import init_test_distributed_environment
        init_test_distributed_environment(1, 1, 0, "29598")
    if use_kpack:
        monkeypatch.setenv("VLLM_RDNA2_W4_KPACK", "1")
    else:
        monkeypatch.setenv("VLLM_RDNA2_W4_KPACK", "0")

    cfg = MPLinearLayerConfig(
        full_weight_shape=(K, N),
        partition_weight_shape=(K, N),
        weight_type=scalar_types.uint4,
        act_type=torch.float16,
        group_size=GS,
        zero_points=True,
        has_g_idx=False,
    )
    kern = RDNA2TritonW4A16LinearKernel(
        cfg, "weight_packed", "weight_scale", "weight_zero_point"
    )

    # checkpoint tensors (compressed-tensors orientation)
    shifts = torch.arange(8, device="cuda", dtype=torch.int32) * 4
    w4 = torch.randint(0, 16, (N, K), dtype=torch.int32, device="cuda")
    w_ckpt = torch.sum(
        (w4.view(N, K // 8, 8) & 0xF) << shifts.view(1, 1, 8), dim=2
    ).to(torch.int32)  # [N, K//8] packed along K
    zn = torch.randint(0, 16, (N, K // GS), dtype=torch.int32, device="cuda")
    z_ckpt = torch.sum(
        (zn.view(N // 8, 8, K // GS) & 0xF) << shifts.view(1, 8, 1), dim=1
    ).to(torch.int32)  # [N//8, K//G] packed along N
    s_ckpt = (
        torch.rand(N, K // GS, dtype=torch.float16, device="cuda") * 0.01
    )

    def _loader(*a, **k):
        return None

    layer = _FakeLayer()
    layer.weight_packed = PackedvLLMParameter(
        data=w_ckpt,
        weight_loader=_loader,
        input_dim=1,
        output_dim=0,
        packed_factor=8,
        packed_dim=1,
    )
    layer.weight_scale = GroupQuantScaleParameter(
        data=s_ckpt, weight_loader=_loader, output_dim=0, input_dim=1
    )
    layer.weight_zero_point = PackedvLLMParameter(
        data=z_ckpt,
        weight_loader=_loader,
        input_dim=1,
        output_dim=0,
        packed_dim=0,
        packed_factor=8,
    )

    kern.process_weights_after_loading(layer)

    x = torch.randn(1, K, dtype=torch.float16, device="cuda") * 0.1
    out = kern.apply_weights(layer, x)[0].float()

    z_nk = zn.float().repeat_interleave(GS, 1)  # [N, K]
    s_nk = s_ckpt.float().repeat_interleave(GS, 1)  # [N, K]
    ref = x[0].float() @ ((w4.float() - z_nk) * s_nk).T
    torch.testing.assert_close(out, ref, atol=2e-1, rtol=2e-2)
