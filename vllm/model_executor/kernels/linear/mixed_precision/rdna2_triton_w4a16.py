# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RDNA2 (gfx1030) W4A16 kernel for AWQ/GPTQ using Triton.

RDNA2 lacks v_pk_fma_f16, v_dot2_f32_f16, and v_dot2_f32_bf16, so the
Triton kernel's fp32-accumulator path is used. This handles both
uint4 (AWQ, asymmetric with explicit zeros) and uint4b8 (GPTQ,
symmetric with bias=8).

Weight layout (post-process_weights_after_loading):
  qweight: [K, N//8]  int32  — GPTQ sequential packing (shifts [0,4,...,28])
  scales:  [K//G, N]  fp16
  qzeros:  [K//G, N//8]  int32  (None for symmetric uint4b8)
"""

import torch

from vllm.model_executor.layers.quantization.utils import replace_parameter
from vllm.model_executor.parameter import BasevLLMParameter, permute_param_layout_
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types
from vllm.triton_utils import tl, triton

from .MPLinearKernel import MPLinearKernel, MPLinearLayerConfig
from .triton_w4a16 import triton_w4a16_gemm


class RDNA2TritonW4A16LinearKernel(MPLinearKernel):
    """RDNA2-optimized Triton W4A16 kernel for AWQ and GPTQ."""

    SUPPORTED_QUANT_TYPES = [scalar_types.uint4, scalar_types.uint4b8]

    @classmethod
    def get_min_capability(cls) -> int:
        return 60

    @classmethod
    def can_implement(cls, c: MPLinearLayerConfig) -> tuple[bool, str | None]:
        if not current_platform.is_rocm():
            return False, "RDNA2 Triton W4A16 kernel is ROCm-only"

        from vllm.platforms.rocm import on_gfx10x

        if not on_gfx10x():
            return False, "RDNA2 Triton W4A16 kernel requires gfx1030"

        if c.weight_type not in cls.SUPPORTED_QUANT_TYPES:
            return (
                False,
                f"Quant type ({c.weight_type}) not supported by "
                f"RDNA2 Triton W4A16 kernel; supported: {cls.SUPPORTED_QUANT_TYPES}",
            )

        if c.act_type not in (torch.float16, torch.bfloat16):
            return False, "Only float16/bfloat16 activations are supported"

        N = c.partition_weight_shape[1]
        if N % 8 != 0:
            return (
                False,
                f"Output features ({N}) must be divisible by 8 "
                "(8 int4 values packed per int32)",
            )

        if c.has_g_idx:
            return (
                False,
                "Activation reordering (g_idx) is not supported by "
                "RDNA2 Triton W4A16 kernel",
            )

        gs = c.group_size
        if gs not in [-1, 32, 64, 128, 256] and gs != c.full_weight_shape[0]:
            return (
                False,
                f"Group size {gs} not supported; "
                f"supported: [-1, 32, 64, 128, 256] "
                f"or full K ({c.full_weight_shape[0]})",
            )

        K = c.partition_weight_shape[0]
        eff_gs = gs if gs != -1 else K
        if K % eff_gs != 0:
            return (False, f"Input features {K} not divisible by group size {eff_gs}")

        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Repack weights from checkpoint layout to kernel layout [K, N//8]."""

        def repack_w_q(x: BasevLLMParameter) -> BasevLLMParameter:
            permute_param_layout_(x, input_dim=1, output_dim=0, packed_dim=1)
            w = x.data  # [N, K//8] int32

            N_dim, K8 = w.shape
            K_dim = K8 * 8
            shifts = torch.arange(8, device=w.device, dtype=torch.int32) * 4
            w_unpacked = ((w.unsqueeze(-1) >> shifts) & 0xF).reshape(N_dim, K_dim)
            w_KN = w_unpacked.t().contiguous()
            N8 = N_dim // 8
            w_repacked = torch.sum(
                (w_KN.view(K_dim, N8, 8) & 0xF) << shifts,
                dim=2,
                dtype=torch.int32,
            )
            x.data = w_repacked.contiguous()
            return x

        def repack_w_s(x: BasevLLMParameter) -> BasevLLMParameter:
            permute_param_layout_(x, input_dim=1, output_dim=0)
            x.data = x.data.t().contiguous()
            return x

        self._transform_param(layer, self.w_q_name, repack_w_q)
        self._transform_param(layer, self.w_s_name, repack_w_s)

        if self.w_zp_name is not None:
            zp = getattr(layer, self.w_zp_name, None)
            if zp is not None:
                replace_parameter(
                    layer,
                    self.w_zp_name,
                    torch.nn.Parameter(zp.data.t().contiguous(), requires_grad=False),
                )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        c = self.config
        w_q, w_s, w_zp, _ = self._get_weight_params(layer)

        x_2d = x.reshape(-1, x.shape[-1]).contiguous()
        out_shape = x.shape[:-1] + (c.partition_weight_shape[1],)

        K = c.partition_weight_shape[0]
        group_size = c.group_size if c.group_size != -1 else K

        zp_bias = c.weight_type.bias if c.weight_type.has_bias() else 0

        output = triton_w4a16_gemm(
            a=x_2d,
            b_q=w_q,
            scales=w_s,
            qzeros=w_zp,
            group_size=group_size,
            zp_bias=zp_bias,
        )

        if bias is not None:
            output.add_(bias)

        return output.reshape(out_shape)
