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
from vllm.utils.torch_utils import direct_register_custom_op

from .MPLinearKernel import MPLinearKernel, MPLinearLayerConfig
from .triton_w4a16 import triton_w4a16_gemm


def _rdna2_w4a16_linear_impl(
    x: torch.Tensor,
    w_q: torch.Tensor,
    w_s: torch.Tensor,
    w_zp: torch.Tensor | None,
    bias: torch.Tensor | None,
    group_size: int,
    zp_bias: int,
) -> torch.Tensor:
    """Eager dispatch: M=1 decode GEMV vs the generic tl.dot GEMM.

    Wrapped as an opaque custom op so torch.compile/CUDA-graph capture
    executes this with the real runtime M (dynamic-shape guards are not
    re-evaluated at runtime, so a Python ``if M == 1`` at the call site
    would be baked out of the traced graph).
    """
    if (
        x.shape[0] == 1
        and x.shape[1] % 32 == 0
        and group_size in (32, 64, 128, 256)
    ):
        from vllm.model_executor.layers.rdna2_gemm import rdna2_w4a16_gemv

        return rdna2_w4a16_gemv(
            x, w_q, w_s, w_zp, group_size, zp_bias, bias
        ).unsqueeze(0)

    out = triton_w4a16_gemm(x, w_q, w_s, w_zp, group_size, zp_bias)
    if bias is not None:
        out = out + bias
    return out


def _rdna2_w4a16_linear_fake(
    x: torch.Tensor,
    w_q: torch.Tensor,
    w_s: torch.Tensor,
    w_zp: torch.Tensor | None,
    bias: torch.Tensor | None,
    group_size: int,
    zp_bias: int,
) -> torch.Tensor:
    return x.new_empty((x.shape[0], w_q.shape[1] * 8))


direct_register_custom_op(
    op_name="rdna2_w4a16_linear",
    op_func=_rdna2_w4a16_linear_impl,
    fake_impl=_rdna2_w4a16_linear_fake,
)


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

        # Opaque custom op: the M=1 GEMV dispatch must see the real
        # runtime M at capture, not a traced symbolic shape.
        output = torch.ops.vllm.rdna2_w4a16_linear(
            x_2d, w_q, w_s, w_zp, bias, group_size, zp_bias
        )
        return output.reshape(out_shape)
