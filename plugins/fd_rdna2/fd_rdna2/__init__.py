# Copyright (C) 2026 Aron Hsiao
# SPDX-License-Identifier: GPL-3.0-or-later
# Part of vllm-rdna2-recipe: vLLM on the Radeon PRO V620 (Navi 21 / gfx1030).
"""WS1 flash-decode attention for gfx1030 — vLLM general plugin.

vllm-rdna2 fork adaptation (2026-08-21). Original plugin targeted vLLM 0.27.1
with a combined int8 KV buffer (K and V packed tightly with inline scales)
and a single-byte-sliced i32 read. Our 0.26.x tree splits K and V into
separate tensors and exposes per-token-head scales as dedicated float32
cache tensors, so this port:
  * delegates the int8 decode fast path to our in-tree Apache-2.0 kernel
    (vllm/model_executor/layers/attention/rdna2_fd_decode.py)
  * gates on q.shape[0] == 1 only — speculative verification (q > 1) still
    delegates to stock until a batched multi-query kernel lands
  * reads scales from k_scale_cache / v_scale_cache (separate float32 tensors)
    rather than the recipe's inline-scales byte-sliced view

Enable with FD_RDNA2=1. Disable by unsetting the env var.
"""
import os

_stats = {"fast": 0, "slow": 0, "reason": {}}
_log = None


def _reject(why):
    _stats["slow"] += 1
    n = _stats["reason"].get(why, 0) + 1
    _stats["reason"][why] = n
    if _log is not None and n == 1:
        _log.info("fd_rdna2: delegating to stock kernel (%s)", why)
    return False


def register():
    if os.getenv("FD_RDNA2", "0") != "1":
        return
    import torch
    from vllm.logger import init_logger
    import vllm.v1.attention.backends.triton_attn as ta
    from vllm.v1.kv_cache_interface import KVQuantMode
    from vllm.model_executor.layers.attention.rdna2_fd_decode import fd_decode

    global _log
    logger = _log = init_logger("vllm.fd_rdna2")
    if getattr(ta, "_fd_rdna2_patched", False):
        return
    orig = ta.unified_attention

    CHUNK = int(os.getenv("FD_CHUNK", "512"))
    TILE = int(os.getenv("FD_TILE", "32"))
    WARPS = int(os.getenv("FD_WARPS", "8"))

    def patched(**kw):
        q = kw.get("q")
        try:
            # Fork-private: q.shape[0] == 1 only (decode). MTP verification
            # (q > 1) is still delegated to the stock kernel until a batched
            # multi-query kernel lands.
            ok = (
                kw.get("max_seqlen_q") == 1
                and kw.get("kv_quant_mode") == KVQuantMode.INT8_PER_TOKEN_HEAD
                and q is not None and q.dim() == 3
                and q.shape[0] == 1
                and q.shape[2] == 256                             # head_size for Qwen3.8-27B
                and bool(kw.get("causal"))
                and kw.get("sinks") is None
                and kw.get("alibi_slopes") is None
                and not kw.get("softcap")
                and kw.get("window_size", (-1, -1))[0] < 0
                and kw.get("chunk_lookback", -1) < 0
                and kw.get("rswa_window") is None
                and kw.get("mm_prefix_range") is None
                and kw.get("k_scale_cache") is not None
                and kw.get("v_scale_cache") is not None
                and kw.get("k_descale") is None                    # no fp8 quant
                and kw.get("v_descale") is None
                and kw.get("q_descale") is None
            )
        except Exception:
            ok = False
        if not ok:
            _reject("shape/mode")
            return orig(**kw)

        k = kw["k"]
        v = kw["v"]
        # Our 0.26.x tree splits K and V from the combined cache: k and v have
        # shape (num_blocks, block_size, n_kv, padded_hs). Recipe assumed the
        # combined layout (n_kv in dim 2, padded_hs in dim 3) — the same here.
        n_kv = k.shape[2]
        if (k.dtype != torch.int8 or v.dtype != torch.int8
                or k.shape != v.shape
                or k.shape[3] != 260):  # padded_hs = 256 + 4 (sizeof(float32))
            _reject("layout")
            return orig(**kw)

        block_table = kw["block_table"]
        if block_table.dtype != torch.int32:
            _reject("block_table dtype")
            return orig(**kw)

        seqused_k = kw["seqused_k"]
        if seqused_k is None:
            _reject("no seqused_k")
            return orig(**kw)

        bs = k.shape[1]

        try:
            out = fd_decode(
                q, k, v,
                kw["k_scale_cache"], kw["v_scale_cache"],
                block_table, seqused_k,
                float(kw["softmax_scale"]),
                BS=bs, CHUNK=CHUNK, TILE=TILE, num_warps=WARPS,
            )
        except Exception as e:
            _reject(f"kernel_err: {e}")
            return orig(**kw)

        # Write back into the caller's output buffer (shape matches: q has 1 row).
        kw["out"][:1].copy_(out.view(1, *out.shape))
        _stats["fast"] += 1
        n = _stats["reason"].get("fast decode q=1", 0) + 1
        _stats["reason"]["fast decode q=1"] = n
        if n == 1:
            _log.info("fd_rdna2: fast path taken for decode (q=1)")
        return None

    ta.unified_attention = patched
    ta._fd_rdna2_patched = True
    ta._fd_rdna2_stats = _stats
    logger.info("fd_rdna2: flash-decode override active (CHUNK=%d TILE=%d warps=%d); "
                "decode/int8/hs256/batch1 -> ours, all else -> stock", CHUNK, TILE, WARPS)
