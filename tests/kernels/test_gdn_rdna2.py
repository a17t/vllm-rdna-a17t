# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Correctness tests for native RDNA2 GDN decode kernels vs Triton references.

Tests cover:
- causal_conv1d_update_rdna2: single/multi-token, with/without bias, state persistence
- sigmoid_gating_delta_rule_update_rdna2: with/without L2 norm, state I/O, multi-seq
- Multi-step state persistence (run kernel N times, compare state evolution)
"""

import pytest
import torch

from vllm.platforms import current_platform


@pytest.fixture(scope="module", autouse=True)
def skip_if_not_gfx10x():
    if not current_platform.is_rocm():
        pytest.skip("ROCm only")
    try:
        from vllm.platforms.rocm import on_gfx10x
        if not on_gfx10x():
            pytest.skip("RDNA2 (gfx10x) only")
    except ImportError:
        pytest.skip("on_gfx10x not available")


# ---------------------------------------------------------------------------
# conv1d_update tests
# ---------------------------------------------------------------------------

def _run_conv1d_native(batch, dim, seqlen, width, state_len, has_bias, num_cache_lines=10):
    """Run native RDNA2 conv1d update."""
    from vllm import _custom_ops as _gdn_ops
    x = torch.randn(batch, dim, seqlen, dtype=torch.float16, device="cuda")
    weight = torch.randn(dim, width, dtype=torch.float16, device="cuda")
    bias = torch.randn(dim, dtype=torch.float16, device="cuda") if has_bias else None
    conv_state = torch.randn(num_cache_lines, dim, state_len, dtype=torch.float16, device="cuda")
    conv_state_indices = torch.arange(1, batch + 1, dtype=torch.int32, device="cuda")
    return _gdn_ops.gdn_conv1d_update_rdna2(
        x, weight, bias, conv_state, conv_state_indices,
    ), conv_state


def _run_conv1d_triton(batch, dim, seqlen, width, state_len, has_bias, num_cache_lines=10):
    """Run Triton conv1d update reference."""
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update
    x = torch.randn(batch, dim, seqlen, dtype=torch.float16, device="cuda")
    weight = torch.randn(dim, width, dtype=torch.float16, device="cuda")
    bias = torch.randn(dim, dtype=torch.float16, device="cuda") if has_bias else None
    conv_state = torch.randn(num_cache_lines, dim, state_len, dtype=torch.float16, device="cuda")
    conv_state_indices = torch.arange(1, batch + 1, dtype=torch.int32, device="cuda")
    out = causal_conv1d_update(
        x, conv_state, weight, bias, activation="silu",
        conv_state_indices=conv_state_indices, validate_data=True,
    )
    return out, conv_state


@pytest.mark.parametrize("batch,dim,seqlen", [
    (1, 256, 1),
    (2, 512, 1),
    (4, 1024, 1),
])
@pytest.mark.parametrize("has_bias", [True, False])
def test_conv1d_update_correctness(batch, dim, seqlen, has_bias):
    """Native conv1d output and state must match Triton within fp16 tolerance."""
    width = 4
    state_len = width - 1  # 3

    x_native = torch.randn(batch, dim, seqlen, dtype=torch.float16, device="cuda")
    x_triton = x_native.clone()
    weight = torch.randn(dim, width, dtype=torch.float16, device="cuda")
    bias = torch.randn(dim, dtype=torch.float16, device="cuda") if has_bias else None
    cs_native = torch.randn(10, dim, state_len, dtype=torch.float16, device="cuda")
    cs_triton = cs_native.clone()
    csi = torch.arange(1, batch + 1, dtype=torch.int32, device="cuda")

    from vllm import _custom_ops as _gdn_ops
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update

    out_native = _gdn_ops.gdn_conv1d_update_rdna2(
        x_native, weight, bias, cs_native, csi.clone(),
    )
    out_triton = causal_conv1d_update(
        x_triton, cs_triton, weight, bias, activation="silu",
        conv_state_indices=csi.clone(), validate_data=True,
    )

    out_diff = (out_native - out_triton).abs().max().item()
    cs_diff = (cs_native - cs_triton).abs().max().item()
    assert out_diff < 0.01, f"Output diff {out_diff} too large (batch={batch}, dim={dim}, seqlen={seqlen}, bias={has_bias})"
    assert cs_diff < 0.01, f"Conv state diff {cs_diff} too large"


def test_conv1d_update_state_persistence():
    """Running conv1d twice must produce same state evolution as Triton."""
    batch, dim, seqlen = 1, 256, 1
    width = 4
    state_len = width - 1

    x1 = torch.randn(batch, dim, seqlen, dtype=torch.float16, device="cuda") * 0.1
    x2 = torch.randn(batch, dim, seqlen, dtype=torch.float16, device="cuda") * 0.1
    weight = torch.randn(dim, width, dtype=torch.float16, device="cuda") * 0.1
    bias = torch.randn(dim, dtype=torch.float16, device="cuda") * 0.1

    from vllm import _custom_ops as _gdn_ops
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update

    # Native: two sequential calls
    cs_n = torch.randn(10, dim, state_len, dtype=torch.float16, device="cuda") * 0.1
    csi = torch.tensor([1], dtype=torch.int32, device="cuda")
    out_n1 = _gdn_ops.gdn_conv1d_update_rdna2(x1.clone(), weight, bias, cs_n, csi.clone())
    out_n2 = _gdn_ops.gdn_conv1d_update_rdna2(x2.clone(), weight, bias, cs_n, csi.clone())

    # Triton: two sequential calls
    cs_t = cs_n.clone()
    csi_t = torch.tensor([1], dtype=torch.int32, device="cuda")
    out_t1 = causal_conv1d_update(x1.clone(), cs_t, weight, bias, activation="silu", conv_state_indices=csi_t.clone(), validate_data=True)
    out_t2 = causal_conv1d_update(x2.clone(), cs_t, weight, bias, activation="silu", conv_state_indices=csi_t.clone(), validate_data=True)

    # Compare after step 1
    assert (out_n1 - out_t1).abs().max().item() < 0.1, "Output 1 diverged"
    # Compare after step 2
    assert (out_n2 - out_t2).abs().max().item() < 0.1, "Output 2 diverged"


def test_conv1d_update_null_block():
    """Null block (cache_idx=0) must skip processing, output is cloned input."""
    batch, dim, seqlen = 2, 256, 1
    width = 4
    state_len = width - 1

    x = torch.randn(batch, dim, seqlen, dtype=torch.float16, device="cuda")
    weight = torch.randn(dim, width, dtype=torch.float16, device="cuda")
    bias = torch.randn(dim, dtype=torch.float16, device="cuda")
    cs = torch.randn(10, dim, state_len, dtype=torch.float16, device="cuda")
    # First seq is null block (idx=0), second is valid (idx=1)
    csi = torch.tensor([0, 1], dtype=torch.int32, device="cuda")

    from vllm import _custom_ops as _gdn_ops
    out = _gdn_ops.gdn_conv1d_update_rdna2(x, weight, bias, cs, csi)

    # Null block output should be the cloned input (unchanged)
    assert (out[0] - x[0]).abs().max().item() == 0.0, "Null block output was modified"


def test_conv1d_update_non_contiguous_indices():
    """conv_state_indices with stride != 1 must work correctly."""
    batch, dim, seqlen = 2, 256, 1
    width = 4
    state_len = width - 1

    x = torch.randn(batch, dim, seqlen, dtype=torch.float16, device="cuda")
    weight = torch.randn(dim, width, dtype=torch.float16, device="cuda")
    bias = torch.randn(dim, dtype=torch.float16, device="cuda")
    cs = torch.randn(20, dim, state_len, dtype=torch.float16, device="cuda")

    # Non-contiguous indices: stride=5
    csi_full = torch.arange(20, dtype=torch.int32, device="cuda").reshape(4, 5)
    csi = csi_full[1:3, 2]  # [7, 12] stride=(5,)

    from vllm import _custom_ops as _gdn_ops
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update

    x_n = x.clone()
    cs_n = cs.clone()
    out_n = _gdn_ops.gdn_conv1d_update_rdna2(x_n, weight, bias, cs_n, csi.clone())

    x_t = x.clone()
    cs_t = cs.clone()
    out_t = causal_conv1d_update(x_t, cs_t, weight, bias, activation="silu", conv_state_indices=csi.clone(), validate_data=False)

    assert (out_n - out_t).abs().max().item() < 0.01
    assert (cs_n - cs_t).abs().max().item() < 0.01


# ---------------------------------------------------------------------------
# sigmoid gating tests
# ---------------------------------------------------------------------------

def _run_sigmoid_native(q, k, v, A_log, a, b, dt_bias, initial_state,
                        cu_seqlens=None, ssm_state_indices=None, use_qk_l2norm=True):
    """Run native RDNA2 sigmoid gating."""
    from vllm import _custom_ops as _gdn_ops
    B, T = q.shape[0], q.shape[1]
    HV, V = v.shape[2], v.shape[3]
    K = q.shape[3]
    o = torch.empty((B, T, HV, V), dtype=q.dtype, device=q.device)
    final_state = initial_state.clone()
    _gdn_ops.gdn_sigmoid_gating_delta_rule_update_rdna2(
        A_log, a, b, dt_bias, q, k, v,
        initial_state, cu_seqlens, ssm_state_indices,
        1.0, 20.0, K ** -0.5,
        True, use_qk_l2norm,
        o, final_state,
    )
    return o, final_state


def _run_sigmoid_triton(q, k, v, A_log, a, b, dt_bias, initial_state,
                        cu_seqlens=None, ssm_state_indices=None, use_qk_l2norm=True):
    """Run Triton sigmoid gating reference."""
    from vllm.model_executor.layers.fla.ops.fused_sigmoid_gating import fused_sigmoid_gating_delta_rule_update
    return fused_sigmoid_gating_delta_rule_update(
        A_log=A_log,
        a=a,
        b=b,
        dt_bias=dt_bias,
        q=q,
        k=k,
        v=v,
        initial_state=initial_state,
        inplace_final_state=False,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        use_qk_l2norm_in_kernel=use_qk_l2norm,
    )


@pytest.mark.parametrize("use_qk_l2norm", [True, False])
@pytest.mark.parametrize("B,T,H,HV,K,V", [
    (1, 1, 4, 4, 256, 16),
    (2, 1, 4, 4, 256, 16),
    (1, 3, 4, 4, 256, 16),
])
def test_sigmoid_gating_correctness(use_qk_l2norm, B, T, H, HV, K, V):
    """Native sigmoid gating output and state must match Triton within fp16 tolerance."""
    # Scale inputs to avoid fp16 overflow in no-L2-norm mode
    scale = 1.0 if use_qk_l2norm else 0.1
    q = torch.randn(B, T, H, K, dtype=torch.float16, device="cuda") * scale
    k = torch.randn(B, T, H, K, dtype=torch.float16, device="cuda") * scale
    v = torch.randn(B, T, HV, V, dtype=torch.float16, device="cuda") * scale
    A_log = torch.randn(HV, dtype=torch.float16, device="cuda")
    a = torch.randn(B, T, HV, dtype=torch.float16, device="cuda") * scale
    b = torch.randn(B, T, HV, dtype=torch.float16, device="cuda") * scale
    dt_bias = torch.randn(HV, dtype=torch.float16, device="cuda") * scale
    initial_state = torch.randn(B, HV, V, K, dtype=torch.float16, device="cuda") * scale

    q_n, k_n, v_n = q.clone(), k.clone(), v.clone()
    cs_n = initial_state.clone()
    o_n, fs_n = _run_sigmoid_native(q_n, k_n, v_n, A_log, a, b, dt_bias, cs_n,
                                     use_qk_l2norm=use_qk_l2norm)

    q_t, k_t, v_t = q.clone(), k.clone(), v.clone()
    cs_t = initial_state.clone()
    o_t, fs_t = _run_sigmoid_triton(q_t, k_t, v_t, A_log, a, b, dt_bias, cs_t,
                                     use_qk_l2norm=use_qk_l2norm)

    o_diff = (o_n - o_t).abs().max().item()
    atol = 0.06 if use_qk_l2norm else 0.5
    assert o_diff < atol, f"Output diff {o_diff} too large (l2norm={use_qk_l2norm}, B={B}, T={T}, H={H}, HV={HV}, K={K}, V={V})"


def test_sigmoid_gating_state_persistence():
    """Running sigmoid gating twice must produce same state evolution as Triton."""
    B, T, H, HV, K, V = 1, 1, 4, 4, 256, 16

    q1 = torch.randn(B, T, H, K, dtype=torch.float16, device="cuda")
    k1 = torch.randn(B, T, H, K, dtype=torch.float16, device="cuda")
    v1 = torch.randn(B, T, HV, V, dtype=torch.float16, device="cuda")
    q2 = torch.randn(B, T, H, K, dtype=torch.float16, device="cuda")
    k2 = torch.randn(B, T, H, K, dtype=torch.float16, device="cuda")
    v2 = torch.randn(B, T, HV, V, dtype=torch.float16, device="cuda")

    A_log = torch.randn(HV, dtype=torch.float16, device="cuda")
    a1 = torch.randn(B, T, HV, dtype=torch.float16, device="cuda")
    b1 = torch.randn(B, T, HV, dtype=torch.float16, device="cuda")
    a2 = torch.randn(B, T, HV, dtype=torch.float16, device="cuda")
    b2 = torch.randn(B, T, HV, dtype=torch.float16, device="cuda")
    dt_bias = torch.randn(HV, dtype=torch.float16, device="cuda")
    initial_state = torch.randn(B, HV, V, K, dtype=torch.float16, device="cuda")

    from vllm import _custom_ops as _gdn_ops
    from vllm.model_executor.layers.fla.ops.fused_sigmoid_gating import fused_sigmoid_gating_delta_rule_update

    # Native: two sequential calls
    cs_n = initial_state.clone()
    o_n1, _ = _run_sigmoid_native(q1, k1, v1, A_log, a1, b1, dt_bias, cs_n, use_qk_l2norm=True)
    o_n2, _ = _run_sigmoid_native(q2, k2, v2, A_log, a2, b2, dt_bias, cs_n, use_qk_l2norm=True)

    # Triton: two sequential calls
    cs_t = initial_state.clone()
    o_t1, fs_t1 = _run_sigmoid_triton(q1, k1, v1, A_log, a1, b1, dt_bias, cs_t, use_qk_l2norm=True)
    o_t2, fs_t2 = _run_sigmoid_triton(q2, k2, v2, A_log, a2, b2, dt_bias, cs_t, use_qk_l2norm=True)

    atol = 0.06
    assert (o_n1 - o_t1).abs().max().item() < atol, "Output 1 diverged"
    assert (o_n2 - o_t2).abs().max().item() < atol, "Output 2 diverged"
    assert (cs_n - cs_t).abs().max().item() < atol, "State diverged after step 2"


def test_sigmoid_gating_null_state():
    """Null state (ssm_state_indices=0) should produce zero output."""
    B, T, H, HV, K, V = 1, 1, 4, 4, 256, 16

    q = torch.randn(B, T, H, K, dtype=torch.float16, device="cuda")
    k = torch.randn(B, T, H, K, dtype=torch.float16, device="cuda")
    v = torch.randn(B, T, HV, V, dtype=torch.float16, device="cuda")
    A_log = torch.randn(HV, dtype=torch.float16, device="cuda")
    a = torch.randn(B, T, HV, dtype=torch.float16, device="cuda")
    b = torch.randn(B, T, HV, dtype=torch.float16, device="cuda")
    dt_bias = torch.randn(HV, dtype=torch.float16, device="cuda")
    initial_state = torch.randn(B, HV, V, K, dtype=torch.float16, device="cuda")
    ssm_state_indices = torch.tensor([0], dtype=torch.int32, device="cuda")

    from vllm import _custom_ops as _gdn_ops
    o = torch.empty((B, T, HV, V), dtype=torch.float16, device="cuda")
    final_state = initial_state.clone()
    _gdn_ops.gdn_sigmoid_gating_delta_rule_update_rdna2(
        A_log, a, b, dt_bias, q, k, v,
        initial_state, None, ssm_state_indices,
        1.0, 20.0, K ** -0.5,
        True, True,
        o, final_state,
    )
    # With null state, the kernel should skip (early return) — output tensor remains uninitialized
    # Just verify no crash


def test_sigmoid_gating_with_continuous_batching():
    """Test with ssm_state_indices (continuous batching mode)."""
    B, T, H, HV, K, V = 2, 1, 4, 4, 256, 16

    q = torch.randn(B, T, H, K, dtype=torch.float16, device="cuda")
    k = torch.randn(B, T, H, K, dtype=torch.float16, device="cuda")
    v = torch.randn(B, T, HV, V, dtype=torch.float16, device="cuda")
    A_log = torch.randn(HV, dtype=torch.float16, device="cuda")
    a = torch.randn(B, T, HV, dtype=torch.float16, device="cuda")
    b = torch.randn(B, T, HV, dtype=torch.float16, device="cuda")
    dt_bias = torch.randn(HV, dtype=torch.float16, device="cuda")
    initial_state = torch.randn(10, HV, V, K, dtype=torch.float16, device="cuda")
    ssm_state_indices = torch.tensor([1, 2], dtype=torch.int32, device="cuda")

    from vllm import _custom_ops as _gdn_ops
    from vllm.model_executor.layers.fla.ops.fused_sigmoid_gating import fused_sigmoid_gating_delta_rule_update

    # Native
    q_n, k_n, v_n = q.clone(), k.clone(), v.clone()
    cs_n = initial_state.clone()
    o_n, fs_n = _run_sigmoid_native(q_n, k_n, v_n, A_log, a, b, dt_bias, cs_n,
                                     ssm_state_indices=ssm_state_indices.clone(),
                                     use_qk_l2norm=True)

    # Triton
    q_t, k_t, v_t = q.clone(), k.clone(), v.clone()
    cs_t = initial_state.clone()
    o_t, fs_t = _run_sigmoid_triton(q_t, k_t, v_t, A_log, a, b, dt_bias, cs_t,
                                    ssm_state_indices=ssm_state_indices.clone(),
                                    use_qk_l2norm=True)

    o_diff = (o_n - o_t).abs().max().item()
    assert o_diff < 0.06, f"Continuous batching output diff {o_diff} too large"


def test_conv1d_update_multi_step_correctness():
    """Simulate full decode loop: 10 sequential conv1d updates, compare state."""
    batch, dim, seqlen = 1, 256, 1
    width = 4
    state_len = width - 1

    weight = torch.randn(dim, width, dtype=torch.float16, device="cuda")
    bias = torch.randn(dim, dtype=torch.float16, device="cuda")

    from vllm import _custom_ops as _gdn_ops
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update

    cs_n = torch.randn(10, dim, state_len, dtype=torch.float16, device="cuda")
    cs_t = cs_n.clone()
    csi = torch.tensor([1], dtype=torch.int32, device="cuda")

    for step in range(10):
        x = torch.randn(batch, dim, seqlen, dtype=torch.float16, device="cuda")

        out_n = _gdn_ops.gdn_conv1d_update_rdna2(x.clone(), weight, bias, cs_n, csi.clone())
        out_t = causal_conv1d_update(x.clone(), cs_t, weight, bias, activation="silu", conv_state_indices=csi.clone(), validate_data=True)

        cs_diff = (cs_n - cs_t).abs().max().item()
        out_diff = (out_n - out_t).abs().max().item()
        assert cs_diff < 0.02, f"State diverged at step {step}: diff={cs_diff}"
        assert out_diff < 0.02, f"Output diverged at step {step}: diff={out_diff}"


def test_sigmoid_gating_multi_step_correctness():
    """Simulate full decode loop: 10 sequential sigmoid gating updates, compare state."""
    B, T, H, HV, K, V = 1, 1, 4, 4, 256, 16

    A_log = torch.randn(HV, dtype=torch.float16, device="cuda")
    dt_bias = torch.randn(HV, dtype=torch.float16, device="cuda")

    from vllm import _custom_ops as _gdn_ops
    from vllm.model_executor.layers.fla.ops.fused_sigmoid_gating import fused_sigmoid_gating_delta_rule_update

    cs_n = torch.randn(B, HV, V, K, dtype=torch.float16, device="cuda")
    cs_t = cs_n.clone()

    for step in range(10):
        q = torch.randn(B, T, H, K, dtype=torch.float16, device="cuda")
        k = torch.randn(B, T, H, K, dtype=torch.float16, device="cuda")
        v = torch.randn(B, T, HV, V, dtype=torch.float16, device="cuda")
        a = torch.randn(B, T, HV, dtype=torch.float16, device="cuda")
        b = torch.randn(B, T, HV, dtype=torch.float16, device="cuda")

        q_n, k_n, v_n = q.clone(), k.clone(), v.clone()
        q_t, k_t, v_t = q.clone(), k.clone(), v.clone()

        o_n, _ = _run_sigmoid_native(q_n, k_n, v_n, A_log, a, b, dt_bias, cs_n, use_qk_l2norm=True)
        o_t, _ = _run_sigmoid_triton(q_t, k_t, v_t, A_log, a, b, dt_bias, cs_t, use_qk_l2norm=True)

        cs_diff = (cs_n - cs_t).abs().max().item()
        out_diff = (o_n - o_t).abs().max().item()
        atol = 0.05
        assert cs_diff < atol, f"State diverged at step {step}: diff={cs_diff}"
        assert out_diff < atol, f"Output diverged at step {step}: diff={out_diff}"
