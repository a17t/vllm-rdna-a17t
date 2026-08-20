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
| `VLLM_RDNA2_TRITON_ATTN_TUNING` | 1 | Tuned Triton attention launches (warps=4, stages=1, prefill tile 16; ported from gfx906 fork) |
| `VLLM_TRITON_ATTN_NUM_PAR_SOFTMAX_SEGMENTS` | 16 | 3D split-softmax segments (64 helped MTP on gfx906; VRAM cost) |
| `VLLM_TORCH_SDPA_PREFILL` | 0 | Torch SDPA prefill fallback (TRITON_ATTN + ROCM_ATTN) |
| `VLLM_TORCH_SDPA_DECODE` | 0 | Torch SDPA single-token decode fallback (`_MAX_SEQS` gates batch) |
| `VLLM_TORCH_SDPA_MTP_DECODE` | 0 | Torch SDPA for MTP verify/draft multi-token decode |

## gfx906-mobydick port (this round)
Ported from `ai-infos/vllm-gfx906-mobydick` (v0.20.1rc0) and adapted to gfx1030:

**Correctness (arch-neutral):**
- MTP acceptance fix in `triton_unified_attention`: `query_mask_1` now ANDs
  the row bound (`offs_m < BLOCK_Q*num_queries_per_kv`) — padding rows for
  non-pow2 GQA ratios (e.g. 28/4 → qpk=7) previously read the *next*
  q-block's Q and corrupted every token (A/B-verified: max_err 1.4 → 0.002).
- `compute_tile_loop_bounds`: `max_seq_prefix_len = context_len +
  (q_block+1)*BLOCK_Q` and `qpos_hi = qpos_lo + BLOCK_Q - 1` (replaces the
  off-by-`BLOCK_M//qpk` formulas; affects sliding-window pruning).
- 3D split-softmax now serves multi-token decode (spec-decode verify / MTP):
  segment buffers sized `seq_threshold_3D * (1+num_speculative_tokens)`,
  routing guard `max_seqlen_q > max_seqlen_q_3d` instead of `> 1`, plus
  `q_tokens > buffer` overflow guard.

**Stability:**
- NCCL watchdog workarounds at `rocm.py` import (`TORCH_NCCL_BLOCKING_WAIT=1`
  etc.) — prevents `hipErrorCapturedEvent` during CUDA-graph capture with TP.
- `gpu_worker.py` no longer pops `NCCL_ASYNC_ERROR_HANDLING` on ROCm.
- Worker force-exits 5 s after parent death (VRAM release on abrupt kills).

**Attention tuning (gfx10x-gated, env-toggled):**
- Unified attention: `num_warps=4, num_stages=1`, prefill tile ≤ 16.
- ROCM_ATTN decode fallback: same launch kwargs; GQA padding to 16 removed.
- `torch_attention.py`: Torch SDPA prefill/decode/MTP-decode fallback paths
  (off by default; rocBLAS beat Triton prefill on gfx906 — benchmark here).

**GEMMs:**
- `rdna2_skinny_matmul`: Triton tl.dot skinny GEMM for 1 < M ≤ 16 (MTP
  verify / draft projections) in `rocm_unquantized_gemm_impl` — previously
  hit ~9% MBU `F.linear`.
- FP8 emulation (no native FP8): `RDNA2Fp16ActFp8BlockScaledMMKernel` keeps
  activations fp16 (skips per-token-group fp8 quant) and decodes E4M3 weight
  bytes bitwise in-kernel — no `tl.float8*` in IR (the known RDNA2 Triton
  crash). Applies to linear + fused MoE (`FP16_ACT` path). `supports_fp8()`
  now true on gfx10x (experimental). Tuned MoE config lookup normalizes
  V620-class names to `AMD_GFX1030`.

**GDN:**
- fla packed-recurrent + sigmoid-gating kernels: `num_warps=4, num_stages=1`
  on gfx10x (occupancy on 4-SIMD wave32).

**Also available:** speculative-config `use_local_argmax_reduction=True`
(vocab-parallel draft argmax, upstream) cuts draft all-gather on TP=2.

## Qwen3.8-27B-AWQ-INT4 decode investigation (2026-08-17)

`cyankiwi/Qwen3.8-27B-AWQ-INT4` is compressed-tensors W4A16 **group_size=32**
(all Linear; GDN in_proj/out_proj left unquantized). Served with all RDNA2
native paths off it decoded at **2.2 tok/s** (TP2, batch 1). Bottom-up
microbenchmark attribution (per token, per rank):

| component | kernel | time |
|---|---|---|
| MLP int4 x64 | native awq_gemm_rdna2 @gs32 | 265 ms (!) |
| attn int4 x16 | native | 33 ms |
| GDN fp16 proj x48 | F.linear | 24 ms |
| lm_head fp16 | F.linear | 9 ms |
| GDN core (fla packed-recurrent) x48 | | 4 ms |

The int4 GEMV/GEMM at gs=32 ran at **11-24 GB/s effective** (native) /
56-67 GB/s (Triton tl.dot path) — vs 440-1040 GB/s achievable (fp16 GEMV).
Fixes (2.2 -> **~14.5 tok/s**, 6.6x):

1. **`rdna2_w4a16_gemv`** (`layers/rdna2_gemm.py`): dedicated M=1 W4A16
   Triton GEMV over the `triton_w4a16_gemm` packed layouts — interleave
   unpack + per-group affine dequant, ~171 GB/s effective at gs=32
   (~2.6x the tl.dot path).
2. **Opaque dispatch op** `vllm.rdna2_w4a16_linear`
   (`rdna2_triton_w4a16.py`): the M==1 GEMV choice must execute with the
   real runtime M at CUDA-graph capture. A plain Python `if x.shape[0]==1`
   in `apply_weights` is baked out of the traced graph (dynamic-shape
   guards are not re-evaluated), so the dispatch lives inside a registered
   custom op. Gotcha: the op must return 2-D `(1, N)` — a 1-D GEMV output
   trips inductor's `assert_size_stride`.
3. **Native AWQ kernel gated**: `RDNA2AWQLinearKernel.can_implement` now
   respects `VLLM_RDNA2_NATIVE_AWQ` (previously ignored for
   compressed-tensors checkpoints — it silently won the chooser) and
   rejects `group_size=32` (scale-traffic bound; measured 5x slower than
   the Triton path there).
4. **gfx10x compile defaults** (`config/compilation.py`): parallel-linear
   layers enabled as custom ops so their quant apply is dispatched with
   real shapes at capture.

Notes: first-startup costs ~10 min (fla autotune + fresh AOT compile;
cached afterwards — purge `~/.cache/vllm/torch_compile_cache` after
touching traced code). Remaining decode headroom: GDN packed-recurrent
path when `VLLM_RDNA2_NATIVE_GDN=0` is fine, but the native GDN/paged-attn
kernels still corrupt tokens >1 on this checkpoint (first token correct) —
see Phase-4 TODO below; `FP16_GEMV` had no effect on this checkpoint
(GDN projections are unquantized but the lm_head/in_proj dominate was
false — actual fp16 total was only ~33 ms of ~455 ms).

## Round 2 (2026-08-18): decode 16.2 -> 19.1 tok/s (round total 2.2 -> 19.1, 8.7x)

- `VLLM_RDNA2_FP16_GEMV=1` validated on the server: 14.5 -> 16.2 tok/s
  (GDN in_proj/out_proj unquantized projections + lm_head through the
  Triton GEMV; ~1040 GB/s on wide shapes).
- W4A16 GEMV v2: exhaustive launch sweep (BLOCK_N/warps/stages/BLOCK_K)
  found BLOCK_N=32 + BLOCK_K=64 universal winners. BLOCK_K=64 spans two
  gs=32 quant groups, so the kernel now loads both group rows and selects
  per-k with a where() (correctness vs the sweep's single-load variant).
  Deterministic heuristic replaces the autotuner (Triton do_bench flushes
  L2 and picked worse configs; autotune also costs capture time).
  Kernel rates: gate/up 229, down 193, qkv 171, o_proj 184 GB/s effective
  (was 171/116/93/111). Per-step int4 total ~28.6 -> ~18.8 ms.
- MTP speculative decoding: mechanically working (acceptance 1.90 mean at
  k=2, per-position 0.68/0.39; ~1.54 at k=1) BUT output corrupts after the
  first accepted tokens on this GDN-hybrid model (diverges from non-spec
  greedy continuation). High acceptance + divergence implicates the
  target-side GDN state at verify time (state snapshot/rollback keyed by
  num_accepted_tokens). PARKED - needs token-level tracing. Do not enable
  MTP for serving until fixed.
- OPEN BUG (Phase 4, same family): native GDN / paged-attn kernels produce
  correct first token then garbage on Qwen3.8 (NATIVE_GDN=1 /
  NATIVE_PAGED_ATTN=1). Both this and the MTP corruption are GDN
  recurrent-state handling bugs; next session should trace state
  snapshots at decode step boundaries (fla spec kernel state indexing vs
  the gdn_attn metadata builder) and the native kernels' state updates.

## Round 3 (2026-08-18): multi-row W4A16 GEMV + MTP status

- **Attribution (52.3 ms/step at 19.1 tok/s)**: int4 GEMVs ~18.8 ms,
  fp16 projections ~12 ms, fla GDN ~4 ms, TP2 allreduces **4.75 ms**
  (measured: 128 x ~37 us NCCL ops; custom-AR unavailable on this
  platform — one-shot AR would save ~3 ms), remainder ~13 ms attention +
  overhead.
- **Multi-row W4A16 GEMV** (`rdna2_w4a16_gemv_m`, M<=4): spec-decode
  verify batches (M=1+k) previously fell off the M=1 fast path to the
  tl.dot GEMM (~3.5x slower). The M-row kernel shares weight loads across
  rows; sweep winner BN=32/warps=4/BK=64 uniformly (127/169/98/87 us at
  M=3 vs 648/549/275/264 us tl.dot). Dispatched in the opaque op for
  1<=M<=4.
- **MTP measured with fast verify: 25.8-28.1 tok/s** (vs 19.1 non-spec,
  +47%) — but **output is corrupted**: token duplication ("2, 2, 3, 3,
  3333..." for a primes prompt vs a correct non-spec listing). High
  acceptance (1.93) + divergence from true greedy = draft and target
  share the same wrong state. Localization: GDN spec state-slot
  machinery — the fla spec kernel reads per-position state slots
  (`spec_state_indices` + `num_accepted_tokens`); the state
  snapshot/rollback postprocess (`postprocess_mamba_align_gpu`) is only
  wired for `mamba_cache_mode="all"` while we run "none" (prefix caching
  off). Upstream has no e2e GDN+MTP test (test_mtp.py is mock-based).
  Next session: trace state slots across verify steps; try
  `--mamba-cache-mode=align` as a workaround config.
- MTP remains DISABLED in the serve script until this is fixed.

## Round 4 (2026-08-18 evening): MTP corruption isolated to the draft pass

Systematic elimination on the 27B (byte-identical corruption each time):
NOT local-argmax, NOT mamba-cache-mode (none vs align), NOT the upstream
gates/launch fixes (cherry-picked 7f17ea4/0fa5fe4 anyway — correct hardening),
NOT our fla launch tuning (reverted ab58787), NOT mixed batches.

**Key result: ngram speculative decoding is byte-exact vs non-spec greedy**
(primes prompt) — the verification/sampling/KV machinery is CORRECT. The
corruption is specific to the MTP DRAFT pass (draft+target agree on a
poisoned context; period-(k+1) token duplication = state/KV advanced past
accepted positions). Diffed every spec-path file (qwen_gdn_linear_attn,
gdn_attn, step3p5/llm_base proposers, fla kernels) against origin/main:
functionally identical — the bug very likely persists upstream on this
base; upstream meanwhile reworked the path (fused CUDA post-conv MTP
decode kernel #51674 + head-ratio support #52539, both requiring files we
lack).

Recommended path to correct 28 tok/s MTP serving: rebase main+rdna2 onto
current origin/main (~589 commits, absorbs the Qwen3.5 MTP rework), then
re-validate. MTP stays disabled until then.

Also: Qwen3.5-0.8B is unusable as a fast repro vehicle on gfx1030
(fp16 = numeric garbage on this small model, bf16 = no fdot2.bf16 ISA,
fp32 = 192 GiB profiler OOM).

## Round 5 (2026-08-18 night): rebase onto origin/main — perf retained, MTP still open

Rebased main+rdna2 (45 commits) onto current origin/main (~650 upstream
commits, incl. the Qwen3.5 MTP rework, mamba state-copy race fix, MTP
all-reduce fusion). Build: `HIP_CLANG_PATH=/opt/rocm/lib/llvm/22/bin/
TORCH_ROCM_ARCH=gfx1030 python setup.py develop`. Only 3 conflicts (envs,
qwen_gdn_linear_attn, auto_awq — all "keep both sides"); 2 of our commits
auto-dropped (already upstream).

Validation:
- unit suites: 264/265 (1 known parallel-load flake in test_gdn_rdna2,
  passes standalone); MTP-attention regression tests 8/8
- **non-spec server: 19.2 tok/s, output correct — all perf retained**
- kernel selections correct (RDNA2TritonW4A16 + GEMV; upstream fused-CUDA
  GDN kernel auto-falls-back to triton on gfx1030)

MTP after rebase: STILL corrupt (period-3 duplication, acceptance 2.11).
Root cause refined:
1. The upstream rework (fused post-conv MTP decode kernel) is **BF16-only**
   (dtype gate in `_fused_gdn_decode_unsupported_reason`); gfx1030 runs
   fp16 (bf16 lacks fdot2 ISA for the Triton paths), so we can't inherit it.
2. The MTP draft layer is **full_attention** (not GDN) -> the corruption is
   in the TARGET's GDN spec state-slot handling: the fla spec kernel's
   per-position state selection (`i_t = num_accepted_tokens - 1` slot read
   in fused_recurrent/fused_sigmoid_gating) + `spec_state_indices_tensor`
   construction/rollback in gdn_attn.py. ngram (no draft model, same
   verify machinery minus GDN multi-token states) is byte-exact, which
   brackets it precisely.
3. Rebased MTP needs `max_cudagraph_capture_size` capped (e.g. 64): the
   upstream graph-memory estimate for spec+FULL graphs is 11.14 GiB, else
   KV cache sizing fails at 262144 ctx.

Next session: token-trace `spec_state_indices_tensor` + state slots across
verify rounds on the fp16 triton path.

## Round 6 (2026-08-19): HIP K-packed W4A16 GEMV — 19.2 -> 25.2 tok/s

Ported the hand-written HIP M=1 W4A16 dequant-GEMV from
sebastianmechno-sys/vllm-rocm-windows-rdna2 (Apache-2.0; RX-6000 Windows
project — credit for the kernel design and bandwidth strategy).

- **Kernel** (`csrc/rocm/gemv_w4_kpack_rdna2.cu`): K-packed weights
  [K//8, N] (8 consecutive K values per int32, vs our Triton layout's
  N-packing), lane->N thread ownership with COLS=4 (one 16-byte
  dwordx4 per k-row, 512B coalesced), `__builtin_nontemporal_load`
  streaming reads, UNROLL loads in flight, split-K (gridDim.y + fp32
  atomicAdd), direct fp16 store when no split. Symmetric + asymmetric
  (uint8 zeros) variants.
- **Measured (cold L2)**: mlp gate/up 44.4us (596 GB/s!), down 44.9us
  (589), qkv 36.7us (340), o_proj 36.9us (337) — vs the Triton GEMV's
  136/143/81/76us. ~3x on the wide shapes; the wide ones now exceed the
  512 GB/s DRAM spec (streaming loads avoid L2 pollution).
- **Single layout** (`VLLM_RDNA2_W4_KPACK=1`, gfx10x default): weights
  repacked at load into [K//8, N]; zeros unpacked to [K//G, N] uint8.
  M=1 -> HIP GEMV; M>1 (spec verify, prefill) -> new Triton K-packed
  GEMM (`w4a16_kpack_gemm`, interleave unpack along K, tl.dot).
  The opaque dispatch op now carries an explicit `kpack` layout flag
  (baked-out-shape-inference lesson from round 1: never infer layouts
  from tensor shapes inside traced code).
- **Bug found on the way**: the compressed-tensors zp is [N//8, K//G]
  packed along dim-0; unpack-then-transpose is required (transpose-
  then-unpack reads the wrong axis), and all nibble shifts need *4.
  Caught by a new scheme-level roundtrip test
  (tests/kernels/quantization/test_w4a16_scheme_roundtrip.py) which
  builds a real compressed-tensors layer end-to-end — unit tests with
  self-packed tensors missed it because they shared the same packing
  assumption.
- **Server**: 25.2 tok/s validated ("Paris." + warm runs stable),
  +31% over the Triton GEMV path; round total 2.2 -> 25.2 = 11.5x.
  Full RDNA2 suites 280/280.
- int4 GEMV cost/step now ~5.3 ms (was ~18.8); remaining decode time is
  dominated by fp16 projections (~12 ms incl. lm_head ~3 ms at 443
  GB/s), allreduce 4.75 ms, GDN core 4 ms, attention + overhead.

## Round 7 (2026-08-20): MTP root cause found and FIXED (correctness)

The MTP corruption (period-(k+1) duplication from the first rejection,
acceptance ~2.1 with draft+target agreeing on poisoned context) was the
RDNA2 native spec conv kernel lacking rejection anchoring: after a
rejection it left the REJECTED drafts in the conv window, so the next
verify read phantom context (the target's own argmax then re-emitted
previously drafted tokens — matching every observed signature).
Phase-B round tracing proved onset == first rejected round; the fla
ssm kernel was already anchored; ngram never enters the spec GDN path.

Fix: anchored `gdn_conv1d_update_spec_rdna2` (num_accepted_tokens +
max_query_len args; window init at state[a..a+2], a = accepted-1; state
shift a+1 + query tail; effective state_len = width-1+maxq-1 matching
the reference wrapper). Unit-tested 6/6 vs the Triton reference for
accepted in {1,2,3}. Two test-harness gotchas documented in NEXT_STEPS
(reference mutates x in place; x must be [tokens, dim]).

Validated: primes byte-correct, coherent outputs, acceptance 2.80,
non-spec control unaffected (25.16 tok/s). MTP THROUGHPUT still regressed
(2.5 tok/s; ~1.1 s/step; 86% worker time in accepted-counts event sync;
GPU 99% busy — suspected NCCL spin interplay). MTP remains disabled in
the serve script until Item 1b (NEXT_STEPS.md) lands. All RDNA2 suites
green (267/267 affected + full GDN 26/26).

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
