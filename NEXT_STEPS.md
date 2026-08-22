# NEXT STEPS — parked work items (updated 2026-08-22)

## Item 6 (DONE 2026-08-22, DIAGNOSE-ONLY): AWQ+MTP root cause = kpack GEMM at M>1

The entire 10x AWQ MTP regression (2.9-3.2 t/s) is the **Triton
`w4a16_kpack_gemm` kernel used for M>1 calls**. Clean A/B on the identical
bundle (only VLLM_RDNA2_W4_KPACK differs, 5 warm runs each):

| AWQ + bundle + MTP k=2 | short-prompt tok/s |
|---|---:|
| kpack=1 (default) | 2.9-3.2 |
| kpack=0 | **32.0** |

**Mechanism** (rdna2_triton_w4a16.py:54-72): with kpack ON,
- M=1 (plain decode) -> HIP `gemv_w4_kpack` op: fast (~85 us/layer, 580 GB/s).
- M>1 (spec verify M=3, and any small batch) -> `_kpack_gemv_batched` ->
  Triton `w4a16_kpack_gemm`: **NEVER benchmarked at small M** (kpack round-6
  was validated at M=1 only).
- kpack=0: M<=4 -> `rdna2_w4a16_gemv_m` (fast at M=3).

Microbench (real 27B shapes, KxN in {5120x5120, 9216x5120, 5120x9216},
gs=32, cards 0):

| kernel | M=1 | M=3 | M=4 |
|---|---:|---:|---:|
| `w4a16_kpack_gemm` | 1452-2900 us | 1455-2936 us | 1582-2953 us |
| `rdna2_w4a16_gemv_m` | 86-148 us | 98-174 us | 98-130 us |

kpack GEMM is ~0 GB/s effective at these M — 15-17x slower than gemv_m.
~200 W4 layers x ~2.8 ms = ~560 ms/step, matching the observed 3 t/s MTP
regression exactly. GPTQ+MTP was never affected (Exllama has its own M<=8
fast path).

**Greediness**: AWQ kpack=0 + MTP = 7/8 identical + 1 benign single-token
drift ("easy borrowing" -> "borrowing" at char 577 of a 220-tok essay,
re-converges) vs the kpack=1 baseline — expected kernel-swap numerics, not
corruption. Primes remain byte-exact.

**Parked fix (next session, ~1 line + tests)**: in
`_op_dispatch_kpack` (rdna2_triton_w4a16.py:54), route `1 < M <= 4` to
`rdna2_w4a16_gemv_m` when kpack is on (needs the N-packed weight view or a
kpack-aware gemv_m variant — decide after checking how the kpack repack
step stores weights; possibly just unpack on the fly or keep both
layouts). Until then: **AWQ+MTP users must set VLLM_RDNA2_W4_KPACK=0**
(costs ~10% at M=1 decode vs kpack: 32.0 vs 34.7 t/s short-ctx MTP=0
equivalent).

**Also found this round**:
- AWQ bundle MTP=0: 30.1 t/s @3.5k / 26.6 @13.4k (vs GPTQ 34.7/34.1).
  Remaining gap = AWQ's unquantized fp16 GDN+MTP weights (the checkpoint's
  ignore list; ~12 ms/step) — NOT kernel-side.
- AWQ 41k-ctx runs die: chunked-prefill 7760-token chunk takes >3 min
  (eager GDN prefill on fp16 projections + FULL_DECODE_ONLY = no piecewise
  graphs for prefill), exceeding the 300 s engine dequeue timeout. GPTQ
  quantized GDN so its prefill is ~4x smaller. Workarounds: raise
  VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS, use FULL_AND_PIECEWISE for AWQ, or
  cap ctx. Proper fix = quantized-GDN AWQ checkpoint (out of scope,
  checkpoints frozen).

## Item 5 (DONE 2026-08-22): recipe REPLICATED on GPTQ — MTP shipping

`serve_q38_27b_gptq.sh` = the full vllm-rdna2-recipe config on our fork
(btbtyler09/Qwen3.8-27B-GPTQ-4bit). Results (idle host, verify/decode-rate.py):

| ctx | fp16 AWQ old stack | GPTQ bundle MTP=0 | GPTQ bundle MTP=2 | recipe MTP=2 |
|---|---:|---:|---:|---:|
| ~3.5k | 25.7 | 34.7 | **41.3** | 43-51 |
| ~11-14k | 14.2 | 34.1 | **36.6** | 50.4 |
| ~37-41k | 6.3 | 29.9 | **28.5** | 41.4 |

- **MTP k=2 is greedy-lossless (8/8)** via the checkpoint's own head
  (qwen3_5_mtp), acceptance 2.2 mean / 61% draft (recipe: 58%).
- Root cause of our earlier GPTQ garbage: our fork's RDNA2 W4 kernels
  auto-winning dispatch for GPTQ. Fix = `--linear-backend exllama`
  (Exllama was compiled+HIP-ready all along; the recipe's
  VLLM_DISABLED_KERNELS list is a no-op on our tree — it names kernels
  that are already auto-rejected here).
- Recipe patch 0004 (BLOCK_KN_SIZE 256): +4.5% measured.
- fd_rdna2 mq kernel (batched q<=4 verify, one KV pass): numeric 1e-4 vs
  reference; without it MTP regressed at long ctx exactly as the recipe
  warned (26.2 vs 28.5 @37k).
- FULL_DECODE_ONLY: graph capture 13 s vs 434 s (FULL_AND_PIECEWISE at
  131k ctx) — fixes the slow-restart pain too.

**Remaining gaps to recipe (parked)**:
1. **MTP + prefix caching anomaly**: with MTP=2, TTFT at long ctx implies
   FULL re-prefill every request (147 s @37k; MTP=0 same harness: 2.5 s).
   Prefix cache appears ineffective under spec decode on hybrid models —
   worth its own investigation (mamba-align block boundaries + spec?).
2. Mid/long-ctx gap (36.6 vs 50.4 @14k; 28.5 vs 41.4 @41k): candidates =
   TunableOp tuned-CSV (we run untuned), their torch 2.11 gfx1030 build,
   PCIe slot width (recipe insists two x16-rooted cards; our pair 0,1 may
   be x8 — halves P2P for allreduce). Measure in-server comms first.
3. AR_RDNA2 allreduce plugin: vendored+installable, still unvalidated
   (needs the 8/8 greedy gate before enabling in production).

## Item 1 (DONE 2026-08-20): MTP correctness — ROOT CAUSE FOUND & FIXED

**Root cause**: the RDNA2 native spec conv kernel
(`gdn_conv1d_update_spec_rdna2`) had NO rejection anchoring: after a
rejection it left rejected drafts in the conv window (phantom-context
corruption; period-(k+1) duplication from the first rejection; target
argmax emitted previously-drafted tokens). Phase-B/C tracing proved:
corruption onset == first rejected round; the ssm path (fla) was already
anchored correctly; only the conv was wrong.

**Fix**: anchored native kernel (window init at state[a..a+2], a =
num_accepted-1; state shift a+1 with query tail; effective state_len =
width-1+maxq-1) — unit-tested against the Triton reference for accepted
in {1,2,3} (6/6). GOTCHAS that cost hours: the reference runs IN-PLACE
on x, and expects x as [tokens, dim] (strides), so test harnesss must
clone per call and NOT transpose.

**Validated**: primes byte-correct ("2, 3, 5, 7, 11, 13, 17, 19, 23..."),
coherent rivers think-text, acceptance 2.8, non-spec control 25.16 tok/s.

## Item 1b (NEW, ACTIVE 2026-08-21): MTP throughput regression — diagnostic

Correct MTP runs at **3.17 tok/s** (k=2, ~316 ms/tok) vs **20.4 tok/s** nospec
on the same cards 0,1 build (correctness fix `a97bc98864` retained, acceptance
~2.8, byte-exact primes prompt). The pre-fix corrupted run was 25 tok/s but
output garbage; non-spec on cards 2,3 is 25.2 tok/s.

### Diagnostic probe results (2026-08-21, single-prompt greedy, 80 tok, cards 0,1)

| Label | Config | tok/s | sec / step* |
|---|---|---:|---:|
| nospec_control | no MTP, cudagraph | 20.38 | 49 ms |
| baseline | MTP k=2, cudagraph | 3.17 | 316 ms |
| profile_h2_v2 | MTP k=2 + torch.profiler | 3.22 | 311 ms |
| enforce_eager | MTP k=2, NO cudagraph | 3.10 | 323 ms |
| mtp_k1 | MTP **k=1**, cudagraph | 2.24 | 446 ms |

\* sec/step = wall per output token.

**Key observations**:
1. **k=1 is SLOWER than k=2** (2.24 vs 3.17 tok/s). Per-iter cost: ~673 ms (k=1)
   vs ~883 ms (k=2). The bottleneck is dominated by a **fixed per-step overhead
   of ~600 ms** that does NOT scale with k. Pure verify forward on (k+1) tokens
   would only be (k+1) × 49 ms = 147 ms (k=2) or 98 ms (k=1), so the verify
   forward is being inflated ~6× at k=1 and ~6× at k=2. The fixed overhead is
   the suspect.
2. **cudagraph vs eager are equivalent** (3.10 vs 3.17 tok/s). H4 (graph
   padding mismatch) REFUTED. The bottleneck is NOT in graph mechanics.
3. **nospec on cards 0,1 = 20.4 tok/s** (vs 25.2 on cards 2,3). The
   ~20% gap is hardware variance (low-power state). Kernels healthy.

### Hypotheses REFUTED in this round

- **H1 (async scheduling + mamba_cache_mode interaction)**: refuted by code
  reading. The CPU sync at `gpu_model_runner.py:2176` is taken regardless of
  `async_scheduling` when `mamba_cache_mode='align'` (which is auto-set for
  Qwen3.5 hybrid per `vllm/config/interface.py:605`). Tested by code reading
  rather than experiment; the sync itself is ~0 cost in either mode once
  control reaches the next launch. mamba_cache_mode='align' was already on
  (server log: "Mamba cache mode is set to 'align' for Qwen3_5ForConditional
  Generation by default when prefix caching is enabled"), eliminating H5.
- **H4 (CUDAGraph padding forces oversized launch)**: refuted empirically by
  `--enforce-eager` giving the same 3.10 tok/s as 3.17 with graph.
- **H6 (NCCL spin)**: NCCL kernel time in the torch.profiler trace was
  ~3.9 ms × 84 calls = 327 ms across 27 iters = 12 ms/iter; far below the
  600 ms of unexplained overhead. Not the cause.

### Hypotheses remaining (next round's investigation)

- **H7 (verify pass not graph-captured)**: with `dynamic_shapes_config.type=backed`,
  the verify forward shape (1 req × (k+1) tokens) may fall through to eager
  mode while the postprocess + state copy uses the graph. Each iter would
  then pay ~3-4 eager forward costs (no graph launch amortization).
- **H8 (CUDAGraph capture-size mismatch on hybrid)**: with multi-KV-group
  models and cudagraph padding, the captured graph may be sized for max_num_seqs
  but launch with k+1 token positions — the kernel grid launches on a
  shrunken shape but reads padded metadata (KV cache scatter, slot mapping).
- **H9 (per-iter proposer overhead in `Step3p5MTPProposer`)**: the drafter
  rebuilds per-group block tables + slot mappings on every propose call
  (`step3p5.py:102-120`). With k=2 that's 2 drafter forwards ×
  ~100 us metadata = ~200 us/iter — small in isolation but possibly coupled
  to graph capture ranges.

### Method that worked (reusable harness)

- `mtp_throughput_probe.sh` — single-command harness at repo root:
  launches an MTP server on cards 0,1, waits for ready, runs a warmup
  + 80-tok greedy request, captures rocprof/torch.profiler, prints result.
  See `PROFILE=1 ./mtp_throughput_probe.sh <label>` for torch.profiler and
  `ROCPROF=1 ./mtp_throughth_probe.sh <label>` for rocprofv3.
- Probe results for this round: `/tmp/opencode/mtp_probe/{baseline,
  enforce_eager, profile_h2_v2, mtp_k1, nospec_control}/client.log`.
- Trace outputs: `traces/profiler_h2_v2_out_{0,1}.txt` (CPU-side only,
  GPU events opaque to torch.profiler under cudagraph).

### Why rocprofv3 in this round did not work

`ROCPROF=1 ./mtp_throughput_probe.sh rocprof_h6_v2` started rocprofv3
wrapper but the vllm child never produced output past the initial
deprecation warning. After 20 minutes the harness timed out and rocprof
output dir was empty. Likely cause: rocprofv3's runtime instrumentation
hooks conflict with vLLM's multiprocessing TP worker spawn (`vllm/v1/
executor/multiproc_executor.py`). Use `rocprofv3-attach <pid>` against an
already-running vllm worker process next round; that path was not exercised
in this session.

### 4B cross-check (2026-08-21): MTP works correctly AND fast on Qwen3.5-4B-AWQ

To separate model-scale vs MTP-pipeline effects, ran the same probe matrix
on `QuantTrio/Qwen3.5-4B-AWQ` (4B params, TP1, single V620 card, fp16 AWQ,
native MTP weights bundled). Uses the **`qwen3_5_mtp`** speculative path
(different drafter code from Qwen3-Next's `qwen3_next_mtp`).

| Label | Config | tok/s | sec / step* |
|---|---|---:|---:|
| nospec_4b | no MTP, cudagraph | 7.98 | 125 ms |
| mtp_k2_4b | MTP k=2, cudagraph | **53.30** | **19 ms** |
| mtp_k1_4b | MTP k=1, cudagraph | 44.64 | 22 ms |
| eager_4b_k2 | MTP k=2, NO cudagraph | 46.85 | 21 ms |

\* sec/step = wall per output token. Cards 0 only, TP1.

**Critical observations**:
1. **MTP k=2 is 6.7× faster than nospec on 4B** (53.3 vs 8.0 tok/s). On the
   27B TP2 setup, MTP was 6.4× slower than nospec. **The MTP pipeline is
   not the source of the 27B regression** — when the model fits and TP=1,
   MTP *exceeds* the non-spec throughput via speculative acceptance. The
   27B TP2 bottleneck must be in something that's specific to TP>1 + 27B
   (i.e., NCCL allreduce, mamba-cache pressure, or graph-pad mismatch on
   the large capture-size table).
2. **k=1 vs k=2 on 4B**: k=1 (44.6 tok/s) is *slower* than k=2 (53.3),
   mirroring the 27B pattern (k=1=2.24, k=2=3.17). The fixed-overhead-per-
   iteration hypothesis **survives** but it's a *small absolute* overhead
   at 4B (~1-2 ms/step on a 19 ms total step). Per-iter cost on 4B:
   80/44.6 = 1.79 iter × tok; 80/53.3 = 1.50 iter × tok. With acceptance
   ~2.5 for k=2 and ~1.4 for k=1, per-iter cost ~12-13 ms regardless of k.
   That's the **fixed-overhead signature** — but at 4B it's a small
   fraction of the total step (12/19 = 63%), where at 27B TP2 it was
   overwhelming (~600/870 = 69% but on a 870 ms step).
3. **eager is 12% slower than cudagraph at 4B** (46.85 vs 53.30 tok/s).
   On 27B TP2, eager matched cudagraph (3.10 vs 3.17). This is consistent
   with: at 27B TP2 the cudagraph is bottlenecked by something else (so
   graph doesn't help); at 4B TP1 cudagraph is healthy (so graph helps by
   12% as expected).
4. **nospec 4B is suspiciously slow** (7.98 tok/s on a 4B AWQ model that
   should be ~80 tok/s on V620). Likely cause: shared-memory autotune
   failures during the 13-min startup ("Autotuning failed with out of
   resource: shared memory, Required: 73728, Hardware limit: 65536")
   left some kernels in a degraded autotune config. MTP runs after the
   same warmup finished and shows 53 tok/s, so the warmup completes in
   time for the decode — but the **first** decode (nospec) may have paid
   a JIT-compile cost on a kernel the second run didn't. **The nospec
   baseline on 4B is unreliable**; treat the MTP numbers as the headline.

### Cross-implication for the 27B TP2 regression

The 600 ms fixed per-iter overhead at 27B TP2 is **NOT present** at 4B
TP1 (where MTP is faster than nospec). The cause must therefore lie in
something that scales with model size or TP>1:
- **NCCL allreduce** (H6): 27B TP2 has ~128 NCCL calls × 37 us = 4.75 ms,
  but at TP1 there are zero. Inconsistent with the 600 ms magnitude.
- **CUDAGraph capture-size mismatch** (H8): at 4B TP1 the cudagraph
  memory is 2.58 GiB (vs 11.14 GiB at 27B TP2) and the largest capture
  size is 256 (vs 1024). The graph is much less padded and works fine.
  At 27B TP2, max_num_seqs=64 vs graph capture-size up to 1024 = **16×
  padding**. This is the most plausible remaining cause.
- **mamba-cache allocation pressure**: at 4B TP1 we got the full 71+
  blocks; at 27B TP2 we were capped at 71 blocks. Same constraint,
  similar pressure on both setups, but the **graph** allocation may be
  what interacts with it differently.

### Why rocprofv3 again failed this round

`ROCPROF=1 ./mtp_throughput_probe.sh` on the 4B single-process EngineCore
**still hangs the vllm startup** (server log shows only `rocprofv3 tool
initialization`, no further output). The issue is reproducible on TP1
with no Worker_TP forks — so the prior hypothesis (multiprocess
conflict) is **not** the cause. Suspected: the rocprofv3 launch wrapper
calls `libtorch_compiler_register_attach` or similar initialization
that doesn't terminate cleanly when the torch HIP backend is initialized
inside the same process.

`rocprofv3 --pid <ENGINE_PID>` (attach mode) failed differently: it
attempts to dlopen `librocprofiler-register.so`, which is **not shipped
in this ROCm 7.2 install** (replaced by `librocprofiler-sdk-attach.so`,
a different mechanism using `rocprofiler_attach_set_api_table`).
Stubbing the missing library (`rocprofiler_register_attach` → return 0)
gets past the dlopen but the attach still fails because the real attach
mechanism uses the modern api_table path, not the legacy `register_attach`.

**Recommendation for next session**: file a ROCm bug against this
install (missing `librocprofiler-register.so` + missing
`librocprofiler-register.so.1`) and retry once a corrected ROCm 7.2
package lands. In the meantime, the CPU-side torch.profiler (already
captured) is the only kernel-level signal we have; its limitation
(opaque cudagraph replay) can be worked around with `--enforce-eager`
when MTP correctness is fully validated on the small model.

MTP stays OFF in `serve_q38_27b.sh` until this lands.

### vllm-rdna2-recipe incorporation (2026-08-21, later): fd kernel landed, MTP regression NOT attention

Incorporated leapdragon/vllm-rdna2-recipe (same V620/gfx1030 hardware, same
Qwen3.8-27B arch; GPL-3.0+, vendored under `plugins/` + `verify/` with
`NOTICE-GPL.md` — never upstream those dirs):

1. **0003 port** (our own Apache code, `triton_attn.py`): scale
   `num_par_softmax_segments` on gfx10x so single-seq decode reaches
   MIN_LAUNCH_GRID_SIZE_2D, cap 64. Recipe measured 1.6x at 41k.
2. **`fd_decode` kernel** (in-tree Apache,
   `vllm/model_executor/layers/attention/rdna2_fd_decode.py`): int8
   flash-decode for OUR layout (split K/V views + dedicated f32 scale
   caches; not the recipe's combined buffer). Byte-stripe dots via int32
   views, graph-safe (grid sized by max_ctx, seq read on device).
   **Unit-proven 4/4** (`tests/kernels/test_fd_decode_rdna2.py`, max abs
   diff ~1e-4 vs pure-torch reference). q==1 only; the batched multi-query
   (MTP verify) port is deferred — reference kept at
   `plugins/fd_rdna2/fd_rdna2/fd_kernel2.py`.
3. **fd_rdna2 plugin** (GPL, adapted): gates q==1 + INT8_PER_TOKEN_HEAD +
   hs256 -> our kernel; all else -> stock. Requires
   `--kv-cache-dtype int8_per_token_head`.

**Results (AWQ-INT4, TP2, cards 0,1, MAXSEQS=8, primes byte-correct all):**

| Config | KV | fd | tok/s (80-tok probe) |
|---|---|---|---:|
| nospec | fp16 | - | 20.25 |
| nospec | int8 | no | 22.64 |
| nospec | int8 | yes | 22.15 |
| MTP k=2 | fp16 | - | 3.17 |
| MTP k=2 | int8 | no | 3.21 |
| MTP k=2 | int8 | yes | 3.21 |

Honest decode-only rates (recipe's `verify/decode-rate.py`, excludes
prefill, MAX_LEN 12288): int8+fd **30.6 t/s @3.7k / 29.9 @7.1k**; int8
no-fd **30.9 / 30.1** — **fd is perf-neutral on top of the 0003 patch at
<=8k ctx** (patched stock already fills the GPU; recipe's fd win was
against their UNPATCHED stock at 41k).

**Conclusions**:
- int8 KV: +12% nospec. Correct on AWQ (primes byte-exact).
- The MTP ~10x regression is **NOT attention slope** (nospec decode at 7k
  ctx is still ~30 t/s; attention cannot cost ~670 ms/step). Also NOT KV
  dtype, NOT fd routing, NOT graph-vs-eager (3.10 vs 3.17), NOT MAXSEQS
  padding (8 == 64). The per-step overhead lives in our tree's MTP pipeline
  itself. The recipe runs the same MTP head on vLLM **0.27.1** as a WIN
  (+24% @41k) -> **next step: compare against 0.27.1** (side-by-side build
  or rebase) before more micro-investigation on 0.26.1rc1-dev.
- GPTQ on our tree is BROKEN (user test: btbtyler09/Qwen3.8-27B-GPTQ-4bit
  returns exclamation-mark garbage, stock serve config, fp16 KV). GPTQ W4
  dispatch needs validation before the recipe's GPTQ numbers can be
  reproduced; park it.

**Measurement pitfalls learned**:
- The 80-tok probe's `tok/s` INCLUDES prefill — meaningless for context
  slopes. Use `verify/decode-rate.py` (streams, times first->last delta).
- Concurrent GPU work on the shared host contaminates results (one MTP+fd
  run measured 1.63 t/s while cards 2,3 ran GPTQ tests; identical re-run
  gave 3.21). Re-run outliers before believing them.
- PROFILE=1 does NOT force eager (graphs still captured; profiler pauses
  during capture).

## v0.27.1 merge + "lost gains" incident (2026-08-22): RESOLVED, gains intact

Merged tag v0.27.1 (`a500c4f60e`). The merge was a near-no-op: our base was
already past v0.27.1 content (the 0.26.1rc1 string was versioneer's nearest
tag); the entire merge delta is +21 lines in mxfp4.py (Kimi-K3 helper,
unreachable for Qwen/ROCm). Post-merge: unit tests 30/30, MTP byte-correct
at the same 3.2 t/s.

**The "nospec < 2 t/s, gains lost" report was fully environmental** — three
stacked causes, verified on the idle host:
1. **GPU contention**: serve_q38_27b.sh sets no HIP_VISIBLE_DEVICES, so it
   binds to cards 0,1 — the same cards every probe server used during that
   window. Proven 2x+ penalty from sharing (1.63 vs 3.21 t/s identical
   config).
2. **Cold compile cache** for the prod config hash (probe configs have
   different hashes): ~17 min first start.
3. **First-request JIT** at a fresh shape (user's own observation; now
   baked into the probe as a shape-matched WARMUP2 run, reported
   separately).

**Verified recovery on idle host, merged tree, prod-shaped config**
(fp16 KV, 262144 ctx, util 0.98, FULL_AND_PIECEWISE):
- 80-tok probe: WARMUP2 20.2 / RESULT 23.5 t/s, primes byte-correct.
- Honest decode rates (verify/decode-rate.py): **24.0 t/s @3.2k, 23.0 @3.6k,
  16.4 @7.5k** — at/above the 25.2-era baseline band (which was measured at
  short context). Same after a restart (22.8/22.1).

**Restart cost anatomy** (prod config, warm compile cache):
- torch.compile: ~1 s (cache hit).
- **PIECEWISE CUDA-graph capture: 434 s (~7.2 min), 51 shapes, 6.42 GiB —
  paid on EVERY restart** (graph capture is never cached). This is the
  "startup takes ages" pain, not compilation. FULL_DECODE_ONLY (recipe's
  choice) captures far fewer shapes (~1 min class) and is worth a validated
  trial as a future item; do NOT switch while chasing other regressions.

**Standing measurement protocol** (from here on):
- Exclusive GPU windows: measurements on cards 0,1 only while 2,3 idle
  (and vice versa); serve_q38_27b.sh should pin HIP_VISIBLE_DEVICES
  explicitly when run alongside probes.
- Warm up with 2-3 shape-matched throwaway requests before judging speed.
- Compile cache: keep warm; purge ONLY when traced code changes.


### Root-cause bracket (all evidence from this branch)

- ngram speculative = **byte-exact** vs non-spec greedy -> verify/sampler/
  KV machinery is CORRECT.
- MTP = period-(k+1) token duplication ("2, 2, 3, 3, 3, 333333..."),
  acceptance ~2.1 (draft+target agree on poisoned context).
- MTP draft layer is **full_attention** (not GDN!) -> corruption is in the
  **TARGET's GDN spec state handling**, not the draft model.
- Eliminated: local-argmax, mamba-cache-mode (none/align), fla launch
  tuning, mixed-batch gates fix, packed-decode launch fix, rebase onto
  origin/main (checked after Round-5 rebase: still corrupt).
- Upstream's own fix (fused post-conv MTP decode kernel, #51674) is
  **BF16-only** (dtype gate in `_fused_gdn_decode_unsupported_reason`);
  gfx1030 must run fp16 (bf16 lacks fdot2 ISA) -> can't inherit.

### Where to look (in priority order)

1. `vllm/v1/attention/backends/gdn_attn.py` — construction of
   `spec_state_indices_tensor` (per-draft-position state slots) and
   `num_accepted_tokens` derivation (`torch.diff(query_start_loc)`).
2. `vllm/third_party/flash_linear_attention/ops/fused_sigmoid_gating.py`
   and `fused_recurrent.py` — the spec kernels read the state slot at
   `i_t = num_accepted_tokens[i_n] - 1` (fused_recurrent.py ~line 106):
   verify the slot WRITE side (which position's state is persisted after
   the multi-token verify pass) matches the READ side.
3. `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py`
   `_forward_core` spec branch (~line 1500+): the
   `fused_sigmoid_gating_delta_rule_update(..., ssm_state_indices=
   spec_state_indices_tensor, num_accepted_tokens=...)` call.

### Method that works (reuse these)

- Server A/B harness: `/tmp/opencode/perf/launch_detached.sh` +
  `nospec_texts_rb.json` reference texts (3 prompts, greedy, 80 tok);
  compare byte-exact. Regenerate refs after any model-path change.
- Fast iteration: k=1 spec config (fewer moving parts, still corrupt).
- Env: `VLLM_RDNA2_NATIVE_GDN=0 VLLM_RDNA2_NATIVE_AWQ=1
  VLLM_RDNA2_FP16_GEMV=1 VLLM_RDNA2_NATIVE_PAGED_ATTN=0`.
- MTP serve needs `max_cudagraph_capture_size` capped (64) — upstream
  graph-memory estimate for spec+FULL is 11.14 GiB, breaks KV sizing at
  262k ctx otherwise.
- Purge `~/.cache/vllm/torch_compile_cache` after touching traced code.
- Instrument state slots: checksum `ssm_state_indices` + ssm/conv state
  at verify boundaries (worker side; forward hooks DON'T work under
  torch.compile — use the opaque-op boundary or eager-mode replica:
  `enforce_eager=True, mode 0`).

### Acceptance test

MTP k=2 greedy == non-spec greedy, byte-exact, all 3 reference prompts;
then throughput run (expect 25-28 tok/s sustained).

## Item 2 (QUEUED): native GDN / paged-attn garbage on Qwen3.8

`VLLM_RDNA2_NATIVE_GDN=1` / `VLLM_RDNA2_NATIVE_PAGED_ATTN=1`: correct
first token, garbage after. Same state-handling family as Item 1 — fix
Item 1 first, then retest natives (may fix both). Repro: toggle each env
independently on the serve command; unit repro in
`tests/kernels/test_gdn_rdna2.py` / `test_paged_attention_rdna2.py` with
real Qwen3.8 TP2 dims (HK=8, HV=24, dk=dv=128, conv=4) multi-step.

## Item 3 (QUEUED): decode perf backstop items

- TP2 allreduce = 4.75 ms/step (128 NCCL ops x ~37 us, custom-AR
  unavailable): one-shot AR worth ~3 ms.
- W4A16 GEMV round 3: split-K for narrow-N (qkv/o_proj at 150-170 GB/s
  latency-bound), 64-bit packed loads. Int4 total now ~18.8 ms/step.
- DP2 path (`-dp 2` in serve script) never smoke-tested.

## Item 4 (DONE, reference): current validated state

- Base: origin/main + rebased branch (2026-08-18) + round-6 kpack GEMV.
- Build: `HIP_CLANG_PATH=/opt/rocm/lib/llvm/22/bin/ TORCH_ROCM_ARCH=gfx1030
  python setup.py develop` (venv: /mnt/Dev/vllm-rdna2/venv).
- Serving: `serve_q38_27b.sh` = **25.2 tok/s** validated (batch 1, short
  ctx; VLLM_RDNA2_W4_KPACK=1 default-on: HIP K-packed W4A16 GEMV from
  sebastianmechno-sys/vllm-rocm-windows-rdna2, ~3x the Triton GEMV on
  wide shapes).
- Backup branch: `main+rdna2-backup-pre-rebase` (pre-rebase state).
- Remaining decode budget at 25.2 tok/s (~39.7 ms/step): fp16 projections
  ~12 ms, int4 GEMVs ~5.3 ms, allreduce 4.75 ms, GDN ~4 ms, attn+overhead
  remainder. Next perf levers: MTP correctness (Item 1, x1.7 potential),
  one-shot AR (~3 ms), lm_head at 443 GB/s -> 596-class streaming GEMV.
