# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Correctness tests for the RDNA2 native paged attention kernel.

Strategy: compare the native kernel output against a pure-PyTorch reference
implementation for various configurations of GQA ratio, head size, block size,
and sequence lengths.
"""

import pytest
import torch
from vllm._custom_ops import paged_attention_rocm


def make_kv_cache(
    num_blocks: int,
    num_kv_heads: int,
    head_size: int,
    block_size: int,
    dtype: torch.dtype,
    device: torch.device,
):
    """Create dummy KV cache tensors with the expected paged layout."""
    key_cache = torch.empty(
        (num_blocks, num_kv_heads, head_size, block_size), dtype=dtype, device=device
    )
    value_cache = torch.empty(
        (num_blocks, num_kv_heads, head_size, block_size), dtype=dtype, device=device
    )
    key_cache.uniform_(-0.1, 0.1)
    value_cache.uniform_(-0.1, 0.1)
    return key_cache, value_cache


def paged_attention_reference(
    query,
    key_cache,
    value_cache,
    num_kv_heads,
    scale,
    block_tables,
    seq_lens,
    block_size,
    max_seq_len,
):
    """
    Pure-PyTorch reference for paged attention (decode mode, 1 query per seq).

    Layouts:
      query:      [num_seqs, num_heads, head_size]
      key_cache:  [num_blocks, num_kv_heads, head_size, block_size]
      value_cache:[num_blocks, num_kv_heads, head_size, block_size]
      block_tables:[num_seqs, max_num_blocks_per_seq]
      seq_lens:   [num_seqs]
    """
    num_seqs = query.size(0)
    num_heads = query.size(1)
    head_size = query.size(2)
    gqa_ratio = num_heads // num_kv_heads

    output = torch.zeros(
        (num_seqs, num_heads, head_size), dtype=torch.float32, device=query.device
    )

    for seq_idx in range(num_seqs):
        seq_len = seq_lens[seq_idx].item()
        block_table = block_tables[seq_idx]

        # Gather K and V from paged cache
        k_list = []
        v_list = []
        for token_idx in range(seq_len):
            block_idx = token_idx // block_size
            block_offset = token_idx % block_size
            physical_block = block_table[block_idx].item()
            k_list.append(key_cache[physical_block, :, :, block_offset].float())
            v_list.append(value_cache[physical_block, :, :, block_offset].float())

        k = torch.stack(k_list, dim=0)  # [seq_len, num_kv_heads, head_size]
        v = torch.stack(v_list, dim=0)  # [seq_len, num_kv_heads, head_size]

        # Expand KV heads for GQA
        k_expanded = k.repeat_interleave(
            gqa_ratio, dim=1
        )  # [seq_len, num_heads, head_size]
        v_expanded = v.repeat_interleave(
            gqa_ratio, dim=1
        )  # [seq_len, num_heads, head_size]

        q = query[seq_idx].float()  # [num_heads, head_size]

        for h in range(num_heads):
            # Q.K: dot product per token
            logits = torch.mv(k_expanded[:, h], q[h]) * scale
            # Softmax
            attn_weights = torch.softmax(logits, dim=0)
            # S.V
            for he in range(head_size):
                output[seq_idx, h, he] = (attn_weights * v_expanded[:, h, he]).sum()

    return output.to(query.dtype)


@pytest.mark.parametrize("gqa_ratio", [1, 2, 3, 4, 7, 8, 16])
@pytest.mark.parametrize("head_size", [64, 128])
@pytest.mark.parametrize("block_size", [16, 32])
@pytest.mark.parametrize("seq_len", [1, 16, 64, 256, 512, 1024])
def test_rdna2_paged_attention_correctness(
    gqa_ratio,
    head_size,
    block_size,
    seq_len,
):
    """Test RDNA2 native paged attention against PyTorch reference."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    dtype = torch.float16
    device = torch.cuda.current_device()

    num_seqs = 2
    num_kv_heads = 4
    num_heads = num_kv_heads * gqa_ratio
    max_num_blocks = (seq_len + block_size - 1) // block_size + 2
    num_blocks = max_num_blocks * num_seqs

    key_cache, value_cache = make_kv_cache(
        num_blocks, num_kv_heads, head_size, block_size, dtype, device
    )

    query = torch.empty((num_seqs, num_heads, head_size), dtype=dtype, device=device)
    query.uniform_(-0.1, 0.1)

    block_tables = torch.zeros(
        (num_seqs, max_num_blocks), dtype=torch.int32, device=device
    )
    seq_lens = torch.zeros(num_seqs, dtype=torch.int32, device=device)

    for seq_idx in range(num_seqs):
        actual_len = min(seq_len, 256)
        seq_lens[seq_idx] = actual_len
        num_blocks_seq = (actual_len + block_size - 1) // block_size
        for i in range(num_blocks_seq):
            block_tables[seq_idx, i] = seq_idx * max_num_blocks + i

    # Run native kernel
    scale = 1.0 / (head_size**0.5)
    max_seq_len = 256

    num_partitions = (max_seq_len + 255) // 256
    tmp_output = torch.empty(
        (num_seqs, num_heads, num_partitions, head_size), dtype=dtype, device=device
    )
    exp_sums = torch.empty(
        (num_seqs, num_heads, num_partitions), dtype=torch.float32, device=device
    )
    max_logits = torch.empty(
        (num_seqs, num_heads, num_partitions), dtype=torch.float32, device=device
    )
    output = torch.empty((num_seqs, num_heads, head_size), dtype=dtype, device=device)

    paged_attention_rocm(
        output,
        exp_sums,
        max_logits,
        tmp_output,
        query,
        key_cache,
        value_cache,
        num_kv_heads,
        scale,
        block_tables,
        seq_lens,
        None,  # query_start_loc
        block_size,
        max_seq_len,
        None,  # alibi_slopes
        "auto",  # kv_cache_dtype
        torch.empty((0,), dtype=torch.float32, device=device),  # k_scale
        torch.empty((0,), dtype=torch.float32, device=device),  # v_scale
        None,  # fp8_out_scale
        "auto",  # mfma_type
    )

    # Run reference
    ref_output = paged_attention_reference(
        query,
        key_cache,
        value_cache,
        num_kv_heads,
        scale,
        block_tables,
        seq_lens,
        block_size,
        max_seq_len,
    )

    # Compare
    output_f32 = output.to(torch.float32)
    ref_f32 = ref_output.to(torch.float32)

    max_diff = (output_f32 - ref_f32).abs().max().item()
    atol = 0.2  # fp16 scalar-FMA accumulation tolerance

    assert max_diff < atol, (
        f"max_diff={max_diff:.6f} (atol={atol}), "
        f"gqa={gqa_ratio}, head_size={head_size}, "
        f"block_size={block_size}, seq_len={seq_len}"
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_rdna2_paged_attention_dtypes(dtype):
    """Test both fp16 and bf16 dtypes."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    gqa_ratio = 7
    head_size = 128
    block_size = 16
    seq_len = 64

    num_seqs = 2
    num_kv_heads = 4
    num_heads = num_kv_heads * gqa_ratio
    max_num_blocks = (seq_len + block_size - 1) // block_size + 2
    num_blocks = max_num_blocks * num_seqs

    key_cache, value_cache = make_kv_cache(
        num_blocks,
        num_kv_heads,
        head_size,
        block_size,
        dtype,
        torch.cuda.current_device(),
    )

    query = torch.empty(
        (num_seqs, num_heads, head_size),
        dtype=dtype,
        device=torch.cuda.current_device(),
    )
    query.uniform_(-0.1, 0.1)

    block_tables = torch.zeros(
        (num_seqs, max_num_blocks),
        dtype=torch.int32,
        device=torch.cuda.current_device(),
    )
    seq_lens = torch.zeros(
        num_seqs, dtype=torch.int32, device=torch.cuda.current_device()
    )

    for seq_idx in range(num_seqs):
        seq_lens[seq_idx] = seq_len
        num_blocks_seq = (seq_len + block_size - 1) // block_size
        for i in range(num_blocks_seq):
            block_tables[seq_idx, i] = seq_idx * max_num_blocks + i

    scale = 1.0 / (head_size**0.5)
    max_seq_len = 256
    num_partitions = (max_seq_len + 255) // 256

    tmp_output = torch.empty(
        (num_seqs, num_heads, num_partitions, head_size),
        dtype=dtype,
        device=torch.cuda.current_device(),
    )
    exp_sums = torch.empty(
        (num_seqs, num_heads, num_partitions),
        dtype=torch.float32,
        device=torch.cuda.current_device(),
    )
    max_logits = torch.empty(
        (num_seqs, num_heads, num_partitions),
        dtype=torch.float32,
        device=torch.cuda.current_device(),
    )
    output = torch.empty(
        (num_seqs, num_heads, head_size),
        dtype=dtype,
        device=torch.cuda.current_device(),
    )

    paged_attention_rocm(
        output,
        exp_sums,
        max_logits,
        tmp_output,
        query,
        key_cache,
        value_cache,
        num_kv_heads,
        scale,
        block_tables,
        seq_lens,
        None,
        block_size,
        max_seq_len,
        None,
        "auto",
        torch.empty((0,), dtype=torch.float32, device=torch.cuda.current_device()),
        torch.empty((0,), dtype=torch.float32, device=torch.cuda.current_device()),
        None,
        "auto",
    )

    ref_output = paged_attention_reference(
        query,
        key_cache,
        value_cache,
        num_kv_heads,
        scale,
        block_tables,
        seq_lens,
        block_size,
        max_seq_len,
    )

    output_f32 = output.to(torch.float32)
    ref_f32 = ref_output.to(torch.float32)

    max_diff = (output_f32 - ref_f32).abs().max().item()
    atol = 0.05 if dtype == torch.float16 else 0.1

    assert max_diff < atol, f"max_diff={max_diff:.6f} (atol={atol}), dtype={dtype}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
