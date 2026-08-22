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
    # Batched multi-query (MTP verification) kernels from the original
    # recipe, GPL like this file. They read the COMBINED packed row
    # ([K|kscale|V|vscale], V at +65 i32) — which is exactly this tree's
    # packed-cache layout when the i32 view is anchored at K's section
    # base (k_view.storage_offset()//4), so they run unmodified.
    from .fd_kernel2 import fd2_decode_mq, permute_q_mq

    global _log
    logger = _log = init_logger("vllm.fd_rdna2")
    if getattr(ta, "_fd_rdna2_patched", False):
        return
    orig = ta.unified_attention

    CHUNK = int(os.getenv("FD_CHUNK", "512"))
    TILE = int(os.getenv("FD_TILE", "32"))
    WARPS = int(os.getenv("FD_WARPS", "8"))
    MAXQ = int(os.getenv("FD_MAXQ", "4"))   # batched path packs nq*PAD cols into NQP=32, so 4 max
    _ws = {}

    def patched(**kw):
        q = kw.get("q")
        try:
            # q == 1 is plain decode; q in (1, MAXQ] is a speculative
            # verification pass (MTP drafts + 1 bonus token). nq > 1 rides
            # the batched multi-query kernel — one KV pass for all positions
            # (the recipe's fix for MTP measuring as a long-context loss).
            ok = (
                1 <= (kw.get("max_seqlen_q") or 0) <= MAXQ
                and kw.get("kv_quant_mode") == KVQuantMode.INT8_PER_TOKEN_HEAD
                and q is not None and q.dim() == 3
                and q.shape[0] == kw.get("max_seqlen_q")          # single sequence
                and q.shape[0] <= MAXQ
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
        gqa = q.shape[1] // n_kv
        if (k.dtype != torch.int8 or v.dtype != torch.int8
                or k.shape != v.shape
                or k.shape[3] != 260                # padded_hs = 256 + 4 (sizeof f32)
                or q.shape[1] != n_kv * gqa
                or gqa not in (1, 2, 4, 6, 8, 12, 16)):  # PAD pow2 cover in mq kernel
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
        nq = q.shape[0]

        if nq > 1:
            # ---- batched multi-query verification (original recipe kernels).
            # Build combined i32/f32 views anchored at K's section base: one
            # packed row is [K 64 i32 | kscale 1 | V 64 i32 | vscale 1], so
            # V sits at +65 and the scales at +64/+129 — exactly the layout
            # fd_kernel2 assumes.
            try:
                st = k.untyped_storage()
                f8 = torch.empty(0, dtype=torch.int8, device=k.device)
                f8.set_(st, 0, (st.nbytes(),))
                off32 = k.storage_offset() // 4
                i32 = f8.view(torch.int32)[off32:]
                f32 = f8.view(torch.float32)[off32:]
                if any(s % 4 for s in (k.stride(0), k.stride(1), k.stride(2))):
                    _reject("stride align")
                    return orig(**kw)
                s_blk, s_tok, s_kvh = (k.stride(0) // 4, k.stride(1) // 4,
                                       k.stride(2) // 4)

                bt = block_table[0]
                max_ctx = bt.shape[-1] * bs
                gchunks = (max_ctx + CHUNK - 1) // CHUNK
                PAD = 8   # GQA=6 padded to 8 (beats 16 by 1.6x per the recipe)
                NQP = 16 if nq == 2 else 32
                key = (gchunks, n_kv)
                buf = _ws.get(key)
                if buf is None:
                    if torch.cuda.is_current_stream_capturing():
                        _reject("capture before warmup")
                        return orig(**kw)
                    dev = q.device
                    m = torch.empty(gchunks * n_kv * NQP, dtype=torch.float32,
                                    device=dev)
                    buf = (
                        m, torch.empty_like(m),
                        torch.empty(gchunks * n_kv * NQP * 256,
                                    dtype=torch.float32, device=dev),
                        torch.empty(4 * q.shape[1], 256, dtype=torch.float16,
                                    device=dev),
                        torch.zeros(n_kv, 4, 64, PAD, dtype=torch.float16,
                                    device=dev),
                        torch.zeros(n_kv, 4, 64, 16, dtype=torch.float16,
                                    device=dev),
                        torch.zeros(n_kv, 4, 64, 32, dtype=torch.float16,
                                    device=dev),
                    )
                    _ws[key] = buf
                    _log.info("fd_rdna2: mq buffers for max_ctx=%d (%d chunks"
                              " of %d)", max_ctx, gchunks, CHUNK)
                m, l, a, out_buf, qbuf8, qbuf_mq16, qbuf_mq32 = buf

                qs = (q * kw["softmax_scale"]).to(torch.float16)  # (nq, heads, 256)
                qp = permute_q_mq(qs, nq, GQA=gqa, KVH=n_kv, PAD=PAD,
                                  out=(qbuf_mq16 if NQP == 16 else qbuf_mq32),
                                  NQP=NQP)
                fd2_decode_mq(qp, i32, f32, bt, seqused_k, gchunks, (m, l, a),
                              out_buf, nq, BS=bs, CHUNK=CHUNK, TILE=TILE,
                              GQA=gqa, KVH=n_kv, PAD=PAD, NQP=NQP,
                              num_warps=WARPS, strides=(s_blk, s_kvh, s_tok))
                kw["out"][:nq].copy_(
                    out_buf[: nq * q.shape[1]].view(nq, q.shape[1], 256))
            except Exception as e:
                _reject(f"mq_kernel_err: {e}")
                return orig(**kw)
            _stats["fast"] += 1
            n = _stats["reason"].get(f"fast batched q={nq}", 0) + 1
            _stats["reason"][f"fast batched q={nq}"] = n
            if n == 1:
                _log.info("fd_rdna2: batched fast path for q=%d (one KV pass)",
                          nq)
            return None

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
