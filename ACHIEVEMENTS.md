# vllm-upstream RDNA2 (gfx1030) — Achievements Summary

**Branch:** `main+rdna2` off upstream `main` (post-v0.26.1rc0), later merged v0.27.1
**Hardware:** 2× Radeon Pro V620 (gfx1030, 32 GB, ~512 GB/s) — later expanded to 4 cards
**Models:** Qwen3.6-27B-AWQ, Qwen3.8-27B-AWQ-INT4, Qwen3.8-27B-GPTQ, Qwen3.6-35B-A3B (dense + MoE)

## Headline Results

| Milestone | Date | Decode (batch 1) | Speedup |
|---|---|---|---|
| Baseline (original tree) | 2026-08-17 | **2.2 tok/s** | 1× |
| + W4A16 GEMV + opaque dispatch | 2026-08-17 | 14.5 tok/s | 6.6× |
| + FP16 GEMV env validated | 2026-08-18 | 16.2 tok/s | 7.4× |
| + GEMV v2 (dual-group heuristic) | 2026-08-18 | 19.1 tok/s | 8.7× |
| + HIP K-packed W4A16 GEMV | 2026-08-19 | **25.2 tok/s** | 11.5× |
| GPTQ + MTP k=2 (recipe replicated) | 2026-08-22 | **41.3 tok/s** @3.5k ctx | 18.8× |
| 35B-A3B AWQ, c=64 aggregate | 2026-08-22 | **491 tok/s** | — |

## Major Engineering Achievements

### 1. RDNA2 Platform Bring-up

- gfx10x detection (`on_gfx10x`), MPLinear/kernel choosers
- 6 native HIP kernels: AWQ/GPTQ W4A16 GEMM, AWQ MoE, GDN decode, paged attention
- TurboQuant FP8 guards; FP8 emulation (fp16 activations + bitwise E4M3 decode — avoids Triton gfx1030 `tl.float8*` crash)
- Fast-startup and compile-cache fixes

### 2. Custom GEMV Kernels (the performance wins)

- **`rdna2_gemm.py`** — Triton fp16 GEMV for unquantized M=1 projections (cut the 38% bottleneck to ~5.5% of GPU time)
- **Triton W4A16 GEMV** (M=1, multi-row M≤4) over packed-int4 layout with per-group affine dequant; dispatched via opaque custom op `vllm.rdna2_w4a16_linear` so the M==1 branch executes with real runtime M at CUDA-graph capture (dynamic-shape guards are not re-evaluated at runtime)
- **HIP K-packed W4A16 GEMV** (ported from sebastianmechno-sys/vllm-rocm-windows-rdna2): streaming loads, split-K, COLS=4 coalesced dwordx4, `__builtin_nontemporal_load` — **3× faster** than Triton path (~596 GB/s on wide shapes)
- Deterministic launch heuristics (BLOCK_N=32, BLOCK_K=64) after exhaustive sweeps — Triton's `do_bench` picks measurably worse configs
- Effective int4 GEMV bandwidth: 11-24 GB/s (native kernel) → 171-229 GB/s

### 3. gfx906-mobydick Port (adapted to gfx1030)

- MTP attention correctness: `query_mask_1` ANDs the row bound for non-pow2 GQA ratios; `compute_tile_loop_bounds` off-by-fix
- 3D split-softmax extended to multi-token decode (spec-decode verify / MTP) with proper segment sizing and overflow guards
- NCCL watchdog workarounds at `rocm.py` import (`TORCH_NCCL_BLOCKING_WAIT=1`, etc.) — prevents `hipErrorCapturedEvent` during CUDA-graph capture with TP
- Worker force-exit 5 s after parent death (VRAM release on abrupt kills)
- Torch SDPA prefill/decode/MTP-decode fallback paths (rocBLAS beat Triton prefill on gfx906)
- FLA packed-recurrent + sigmoid-gating tuned for gfx10x wave32/SIMD-4 (`num_warps=4, num_stages=1`)

### 4. Spec Decoding (MTP) — Correctness FIXED

- **Root-caused and fixed** MTP GDN spec-conv corruption (period-(k+1) duplication, target argmax emitting previously-drafted tokens) — rejection anchoring in `gdn_conv1d_update_spec_rdna2` (window init at state[a..a+2], state shift a+1 with query tail)
- Bracketed: ngram spec is byte-exact vs non-spec greedy → verify/sampler/KV machinery correct; corruption was specifically the RDNA2 native spec conv path
- **kpack small-M GEMV** — fixed AWQ MTP 10× regression; kpack=0 env var no longer needed
- fp16 dot in kpack GEMM — 2.3-2.6× prefill speedup
- 4B cross-check: MTP k=2 is 6.7× faster than nospec (53.3 vs 8.0 tok/s) → MTP pipeline healthy, 27B TP2 regression is elsewhere

### 5. Recipe Integration (leapdragon/vllm-rdna2-recipe)

- int8 KV cache (+12% nospec, byte-exact)
- `fd_rdna2` flash-decode plugin (int8, batched multi-query for MTP verify, q≤4) — gates int8_per_token_head KV + head_size 256
- Recipe env bundle — **GPTQ 27B with MTP=2: 41.3/36.6/28.5 tok/s @ 3.5k/11k/37k, greedy 8/8**
- AWQ 27B with full bundle (FD + int8 KV + MTP=2): ~32 tok/s decode
- Recipe patch 0003 (BLOCK_KN_SIZE 256) + fp16 dot in kpack GEMM: +4.5%

### 6. 35B-A3B MoE Benchmarking

- TP4 EP0 optimal; EP loses at every concurrency (no all2all fast path with DP=1)
- TP4 > TP2 decisively (c=1 +37%, c=64 +31%)
- AWQ native MoE HIP kernel beats GPTQ 16-21% (TritonWNA16Experts path)
- AWQ MTP=2 single-stream: **64.2 tok/s** best-in-round (+27%)
- Production `serve_q36_35b.sh`: AWQ default, MTP env knob

### 7. Stability & Tooling

- Branch rebased successfully onto current `origin/main` (~650 upstream commits including the Qwen3.5 MTP rework, mamba state-copy race fix, MTP all-reduce fusion) — perf retained
- Measurement protocol: exclusive-host timings, GPU pinning via `HIP_VISIBLE_DEVICES`, shape-matched warmups, `verify/decode-rate.py` for honest decode rates
- Full RDNA2 test suites green (280/280); MTP-attention regression tests 8/8
- New scheme-level roundtrip test for compressed-tensors W4A16 caught a real packing bug

## Int4 GEMV: Hardware Roofline Reached

Four fundamentally different AWQ kernel designs (packed-fp16, larger TILE_K, double-buffered pipeline, direct-global-read) all produced identical 222 µs/call — the W4A16 dequant GEMV on gfx1030 is **raw-bandwidth-bound at ~31% MBU** and cannot be improved by kernel restructuring.

## Theoretical Ceiling

~55–65 tok/s raw decode (weight-bandwidth roofline, 100% MBU). Current ~25 tok/s/req ≈ 13-15% MBU. The remaining ~2-3× gap is systemic — TP allreduce at batch 1, framework launch overhead, GDN linear-attention compute, structural PyTorch dispatch churn — not single-kernel wins.

## Open / Parked Items

1. **MTP + prefix-cache anomaly** — TTFT under MTP implies full re-prefill every request (147-724 s @37-41k on dense 27B); suspected mamba-align block invalidation or spec-metadata breaking prefix matching on hybrid models
2. **Native GDN / paged-attn** — correct first token then garbage on Qwen3.8 (same recurrent-state family as MTP bug; retest pending MTP fix propagation)
3. **AWQ 27B long-ctx prefill** ~12 min @41k — structural: fp16 GDN projections in checkpoint `ignore` list
4. **TP2 allreduce** ~4.75 ms/step (custom-AR unavailable on this platform; one-shot AR would save ~3 ms)
5. **GPTQ MoE expert kernel gap** (35B) — needs gfx1030 tuning or wiring into RDNA2 MoE HIP kernel
6. **Recipe gap** on 27B GPTQ (28.5 vs 41.4 @37k) — candidates: TunableOp tuned CSV (we run untuned), torch 2.11 gfx1030 build, PCIe slot width
7. **AR_RDNA2 allreduce plugin** — vendored, installable, still unvalidated (gated by 8/8 greedy)
8. **MoE+MTP weight-load bug** — fixed later (unquantized-MTP detection from checkpoint metadata + MoeWNA16 dynamic handling)

## Key Technical Insights (Don't Re-Lose)

- Custom-op dispatch inside `apply_weights` is **not** re-evaluated under CUDA-graph capture → use registered ops with shape/layout flags
- Deterministic launch heuristics beat autotuner (do_bench flushes L2 + costs capture time)
- gfx1030 fp16 works; bf16 lacks `fdot2.bf16` ISA for the Triton paths; fp32 can't be used in profiling (192 GiB OOM)
- GPTQ dense layers on this tree REQUIRE `--linear-backend exllama`; RDNA2 W4 dense kernel produces garbage on GPTQ
- `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800` needed for AWQ long-prefill runs
- rocprofv3 attach is broken on this ROCm 7.2 install (missing `librocprofiler-register.so`) — use torch.profiler CPU-side only
- FP8 emulation strategy: keep activations fp16, decode E4M3 weight bytes bitwise in-kernel — no `tl.float8*` in IR (the known RDNA2 Triton crash)

## External Code & Attribution

The fork incorporates code from three external sources, each with distinct licensing terms. Upstream-PR readiness requires the Apache-2.0-compatible subset only.

### 1. `ai-infos/vllm-gfx906-mobydick` (Apache-2.0) — freely merged

**Base:** `ai-infos/vllm-gfx906-mobydick` v0.20.1rc0, adapted to gfx1030.

| Area | Files / Effect |
|---|---|
| MTP attention correctness | `vllm/v1/attention/ops/triton_unified_attention.py`, `triton_attention_helpers.py` — `query_mask_1` row-bound fix, `compute_tile_loop_bounds` off-by-fix |
| 3D split-softmax | Extended to multi-token decode (spec-decode verify / MTP) |
| Triton attention launch tuning | `envs.py` — `num_warps=4, num_stages=1`, prefill tile ≤ 16 |
| Torch SDPA fallbacks | `vllm/v1/attention/ops/torch_attention.py` — prefill, decode, MTP-decode paths (`VLLM_TORCH_SDPA_*`) |
| NCCL / graph-capture stability | `platforms/rocm.py`, `v1/worker/gpu_worker.py`, `gpu_model_runner.py` |
| FLA kernels | `vllm/third_party/flash_linear_attention/` (already in upstream; re-tuned for gfx10x wave32) |
| Skinny-GEMM tuning | Reference for `vllm/model_executor/layers/rdna2_gemm.py` |

### 2. `sebastianmechno-sys/vllm-rocm-windows-rdna2` (Apache-2.0) — freely merged

Adapted: **`csrc/rocm/gemv_w4_kpack_rdna2.cu`** — hand-written HIP M=1 W4A16 dequant-GEMV (K-packed layout variant).
- COLS=4 coalesced dwordx4, `__builtin_nontemporal_load` streaming reads, UNROLL, split-K via `gridDim.y + atomicAdd`, direct fp16 store when no split
- Symmetric + asymmetric (uint8 zeros) variants
- ~3× speedup over Triton path; ~596 GB/s measured on wide shapes
- Credit for kernel design + bandwidth strategy preserved in file header

### 3. `leapdragon/vllm-rdna2-recipe` (GPL-3.0-or-later) — vendored, isolated

**Author:** Copyright (C) 2026 Aron Hsiao; **License:** GPL-3.0-or-later.
**Isolation:** per `NOTICE-GPL.md` — kept in separately identified directories so the rest of the fork remains Apache-2.0-compatible. **NOT for upstream contribution without dual-licensing arrangement.**

**Vendored verbatim (GPL):**

| Directory | Contents |
|---|---|
| `plugins/fd_rdna2/` | Flash-decode attention plugin + batched multi-query verification path — `FD_RDNA2=1` env gate |
| `plugins/ar_rdna2/` | Push-based one-shot all-reduce over uncached device staging buffers — `AR_RDNA2=1` env gate; unmodified |
| `verify/` | `validate.py`, `decode-rate.py`, `prefill-rate.py`, `soak.py` — A/B harnesses |

**Clean-room re-implemented in Apache-2.0 (kernel ideas reused, code re-written):**

| File | Notes |
|---|---|
| `vllm/model_executor/layers/attention/rdna2_fd_decode.py` | In-tree Apache-2.0 fd-decode for OUR layout (split K/V + dedicated f32 scale caches, not the recipe's combined buffer); `fd_rdna2/__init__.py` delegates here for the q==1 fast path; original GPL combined-buffer kernel retained at `plugins/fd_rdna2/fd_rdna2/fd_kernel2.py` for reference |
| `vllm/v1/attention/ops/triton_attn.py` | Recipe "patch 0003" — scale `num_par_softmax_segments` for single-seq decode on gfx10x; own Apache code |
| `serve_q38_27b_gptq.sh` + BLOCK_KN_SIZE 256 patch | Recipe reproduction, own code |

### 4. Upstream vLLM (Apache-2.0) — base + absorbed merges

- Base: `main` post-v0.26.1rc0
- Merged tag `v0.27.1` (`a500c4f60e`) — near-no-op (+21 lines in `mxfp4.py`)
- ~650 upstream commits absorbed during the Round-5 rebase (incl. the Qwen3.5 MTP rework, mamba state-copy race fix, MTP all-reduce fusion)
- Standard upstream header on most `vllm/` code: `SPDX-License-Identifier: Apache-2.0` / `SPDX-FileCopyrightText: Copyright contributors to the vLLM project`

### 5. Fork-original RDNA2 work (Apache-2.0)

25-commit RDNA2 fork layer — gfx10x detection (`on_gfx10x`), 6 native HIP kernels (AWQ/GPTQ W4A16 GEMM, AWQ MoE, GDN decode, paged attention), MPLinear choosers, Triton W4A16 GEMV + opaque dispatch, FP8 emulation, recipe-0003 port, etc. — fork-original and Apache-2.0-compatible.

### License-cleanliness Summary

| Source | License | Status |
|---|---|---|
| `ai-infos/vllm-gfx906-mobydick` | Apache-2.0 | Freely merged into vLLM tree |
| `sebastianmechno-sys/vllm-rocm-windows-rdna2` | Apache-2.0 | Freely merged; credit in file header |
| `leapdragon/vllm-rdna2-recipe` (GPL components) | **GPL-3.0-or-later** | Isolated in `plugins/fd_rdna2/`, `plugins/ar_rdna2/`, `verify/` |
| `leapdragon/vllm-rdna2-recipe` (kernel ideas) | — | Clean-room re-implementation in-tree (Apache-2.0) |
| Fork-original RDNA2 work | Apache-2.0 | Original |
| Upstream vLLM | Apache-2.0 | Rebase-merged from `origin/main` |

## Sources

- `RDNA2_SUMMARY.md` — investigation notes, env table, build/test workflow
- `NEXT_STEPS.md` — round-by-round progress log
- `HANDOFF-2026-08-23.md` — state and parked items at session end
- `BM35-RESULTS.md` — Qwen3.6-35B-A3B MoE benchmark
- `README.md` — public-facing summary
- `NOTICE-GPL.md` — GPL vendoring attribution
