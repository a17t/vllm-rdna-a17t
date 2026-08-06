# RDNA2 (gfx1030) vLLM Branch — Summary

Branch: `main+rdna2` off upstream `main` (`e98a8774c`, post-v0.26.1rc0).
Tested model: `QuantTrio/Qwen3.6-27B-AWQ` (Qwen3_5, dense, GDN linear attention).
Hardware: 2× Radeon Pro V620 (gfx1030, 32 GB, ~512 GB/s each), TP=2.

## Result
**Decode throughput: ~9 → ~24 tok/s (~2.5×)** on TP=2 with spec decoding.

The speedup came from **one change**: the RDNA2 fp16 GEMV (`rdna2_gemm.py`),
which attacked the dominant decode bottleneck — the unquantized fp16 projections
(GDN `in_proj_a/b`, attention `q/k/v_proj`, layer 0, MTP — excluded from AWQ).
These were running through `torch.nn.functional.linear` at ~9% MBU because
vLLM's skinny-GEMM dispatch excluded gfx10x. The fp16 GEMV (a Triton kernel
lowered to gfx10-safe FMA) cut that 38% bottleneck to ~5.5% of GPU time.

## Working serve config
```bash
VLLM_RDNA2_NATIVE_GDN=1 VLLM_RDNA2_NATIVE_AWQ=1 \
VLLM_RDNA2_FP16_GEMV=1 VLLM_RDNA2_NATIVE_PAGED_ATTN=1 \
vllm serve <model> --dtype float16 --tensor-parallel-size 2 \
  --max-model-len 262144 --reasoning-parser qwen3 \
  --compilation-config '{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}' \
  --language-model-only --gpu-memory-utilization 0.98
```

- `--gpu-memory-utilization 0.98` is required (default 0.90 leaves too little
  headroom for compile + cudagraph + the model on these cards).
- TP=2 is required (TP=1 OOMs: 27B AWQ + mode-3 compile + cudagraph doesn't
  fit on one 32 GB card).

## Env toggles (all default ON except AWQ)
| Env | Default | Effect |
|---|---|---|
| `VLLM_RDNA2_FP16_GEMV` | 1 | Triton fp16 GEMV for M=1 unquantized projections (**the speedup**) |
| `VLLM_RDNA2_NATIVE_GDN` | 1 | Native GDN decode kernels (`gdn_conv1d_update_rdna2` etc.) |
| `VLLM_RDNA2_NATIVE_AWQ` | 0 | Native `awq_gemm_rdna2` via MPLinear chooser (off: raises load memory, no speed gain — AWQ isn't the bottleneck) |
| `VLLM_RDNA2_NATIVE_PAGED_ATTN` | 1 | Native RDNA2 paged attention for full_attention layers |

## What's in the branch (25 commits)
**Ported from the v0.23 fork (Phases 0–7):**
- Platform detection (`on_gfx10x`), 6 HIP kernels (AWQ/GPTQ W4A16 GEMM, AWQ
  MoE, GDN decode, paged attention), MPLinear kernels, MoE methods, GDN
  dispatch, turboquant FP8 guards, config/env tweaks, 4 kernel tests.

**Performance work:**
- `rdna2_gemm.py` — Triton fp16 GEMV for unquantized projections (the win).
- AWQ GEMV: split-K, packed-fp16, double-buffered pipeline, direct-global-read
  (all explored; the AWQ GEMV is at the hardware's raw bandwidth limit — 222 µs
  /call, ~31% MBU, unresponsive to every kernel structure change).

**Memory fixes:**
- Tiled `_convert_awq_to_standard_format` and `RDNA2AWQLinearKernel` repack
  (peaks bounded from ~1.1 GB to ~110 MB per layer).
- `VLLM_RDNA2_NATIVE_AWQ` defaults off (the repack adds load memory and AWQ
  isn't the bottleneck).

## Theoretical ceiling
~55–65 tok/s raw decode (weight-bandwidth roofline, 100% MBU). Currently at
~12 tok/s/req raw (~13–15% MBU). The remaining ~2–3× gap is systemic:
TP all-reduce at batch 1, framework launch overhead, GDN linear-attention
compute, and structural PyTorch dispatch churn — not single-kernel wins.

## AWQ GEMV — what we learned
Four fundamentally different kernel designs all produced identical 222 µs/call:
packed-fp16, larger TILE_K, double-buffered pipeline, direct-global-read.
The W4A16 dequant GEMV on gfx1030 is **raw-bandwidth-bound at ~31% MBU** and
cannot be improved by kernel restructuring. The direct-read variant (no LDS,
no barriers) is kept as the simplest correct implementation.
