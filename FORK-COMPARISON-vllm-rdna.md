# Fork comparison: `vllm-rdna` (rdna_extras) vs our `main+rdna2`

Reference document — 2026-09-03. No code adopted yet; this catalogues what
the other fork has, what we have, and the ranked adoption candidates.

## Fork identity

- **Theirs**: `/mnt/Dev/vllm-rdna2/vllm-rdna`, branch `rdna_extras`, 89
  commits (authors BlivionIaG / kletorch), cherry-picked from an older
  "rdna2_extras" repo onto a **v0.28.0-dev base** (branch point e25c586b9,
  2026-08-23, ~517 upstream commits after our v0.27.1 tag of 2026-08-11;
  the main they track is ~909 commits ahead). Same hardware target
  (gfx1030 / V620), torch 2.12–2.13 + ROCm 7.2 era.
- **Ours**: `vllm-upstream`, `main+rdna2` @ v0.27.1+720 — recipe
  replication stack, MTP machinery, AWQ/kpack kernels.
- Shared lineage: our `gdn_decode_rdna2.cu` and theirs come from the same
  source tree (rdna2_extras).

**API-drift warning**: their ports are v0.28.0-API-shaped (attention
backend API, MoE oracle enum, `vllm/models/deepseek_v4/...` layout).
Cherry-picking onto our v0.27.1 tree needs adaptation — the csrc kernels
are mostly portable, the Python dispatch layers are not drop-in.

## What they have that we don't

| # | Area | Their implementation | Notes / perf claims |
|---|---|---|---|
| 1 | **GDN chunked-prefill in HIP** | `csrc/rocm/gdn_prefill_{prep,kkt,solve_wy,delta_h,o}_rdna2.cu`; dispatch in `layers/mamba/gdn/qwen_gdn_linear_attn.py:94-150` (`_gdn_prefill_chain_rdna2`), auto on gfx10x, no env | Motivated by Triton GDN JIT hangs at 16k ctx / c=8 — cold 16k c=8 completes after port (242 total / 14.3 out tok/s). solve_wy fuses A_inv in ~58KB LDS, killing a ~64MB fp16 round-trip. Parity-tested (prep 27/27, kkt 12/12, solve_wy 22/22) |
| 2 | **W4A16 GPTQ MoE + multi-config prefill** | `csrc/rocm/moe_q_gemm_rdna2.cu` (fused routing+dequant-dot, block_size_m 1/2/4/8), `q_gemm_rdna2_prefill.cu` (V1/A/C tile configs, packed-CAS fp16 atomics); 3-bucket dispatcher in `kernels/linear/mixed_precision/rdna2_w4a16.py:39-55` (exllama for M>256 or wide-N; decode for M≤32 K≥4096; prefill otherwise) | GPTQ-layout (uint4b8) with exllama-identical shuffle, act-order with TP restrictions. **No AWQ kernel in their tree at all** |
| 3 | **TP collectives under torch.compile (real fix)** | Commits 07f1abf47 → 70e65f933: `allow_in_graph` wrappers were never executed by inductor → each rank kept un-reduced partials (silent TP>1 corruption). Fix: plain wrappers traced to `torch.ops.vllm.all_reduce` custom ops + `RocmPlatform.use_custom_op_collectives()=True` (`platforms/rocm.py:1105-1118`) | Working dense+hybrid TP=4 FULL/PIECEWISE; also note their AOT-cache-sharing bug (rank replay) — run `VLLM_USE_AOT_COMPILE=0 VLLM_DISABLE_COMPILE_CACHE=1` on shared caches |
| 4 | **Custom allreduce un-gated (stock kernel)** | `VLLM_FORCE_CUSTOM_ALL_REDUCE` env: forces `fully_connected=True` in `CustomAllreduce.__init__` (`device_communicators/custom_all_reduce.py:179-186`), `use_custom_allreduce()` gate (`platforms/rocm.py:1007-1010`), 5-GPU amdsmi tolerance (`rocm.py:818-828`). **csrc/custom_all_reduce.* untouched** | Claims **+30–34% c=1 on Qwen3.8-27B TP=4**. Tension with our AR_RDNA2 rejection (−12% single-stream): different code (stock vs plugin), different TP/world — the gate-only route on the stock kernel is **untested by us at TP2**; re-test before assuming the recipe's coarse-memory-barrier warning applies |
| 5 | **RDNA_ATTN standalone backend (FA-RDNA2)** | `csrc/rocm/fa_rdna2.cu` (~4.3k lines): decode split-K, prefill, `_short` prefill (N<4k), splitk prefill (N≥4k), **fp8-e4m3 and int8-per-(token,head) KV variants**, head_dim 128+256, sliding window. Backend `vllm/v1/attention/backends/rdna_attn.py`, registry `RDNA_ATTN`, gate `VLLM_USE_RDNA2_FA=1` (default off), priority ahead of ROCM_ATTN on gfx10x | **No q>1**: spec-verify batches deliberately routed to fallback — their note: split-K online-softmax numerics collapse spec acceptance (independently confirms our fd-mq one-pass value) |
| 6 | **MXFP4 W4A4 dense + fused MoE** | `mxfp4_dot2_{dense,moe}.cu` (E2M1 LUT + UE8M0 gs=32, no zeros), `MXF4_RDNA2` oracle routing (`layers/fused_moe/oracle/mxfp4.py`), expert wrapper `experts/rdna2_mxfp4_moe.py` (two-pass w13→SwiGLU→w2, fused topk) | Target: DeepSeek V4 Flash MXFP4 checkpoints (CUTLASS MXFP4 is CDNA-only; Marlin memory-bound at M=1..15) |
| 7 | **DSV4 family: W8A8-FP8 / W8A16(-FP8) / sparse MLA / Lightning Indexer** | `gemm_w8a8_fp8_dense_rdna2.cu`, `moe_w8a16{,_fp8}_rdna2.cu`, `sparse_mla_rdna2.cu` (env `VLLM_USE_RDNA2_MLA=1`, from `vllm/models/deepseek_v4/amd/rocm.py:694`), `indexer_paged_mqa_rdna2.cu` (FP8 paged-MQA; AITER crashes on gfx1030) | "DSV4-Flash decode 3.77×"; indexer 0.58 tok/s @32k ctx PP=5 TP=1 on 5×V620. Model-level integration partially deferred on their side |
| 8 | **EXL3/QTIP stack** (~49 commits) | `exl3_dot2_{dense,moe,dequant}.cu`, `exl3_hadamard.cu` (Sylvester H-128 + suh/svh scales), quant config `layers/quantization/exl3.py`; 3bpw e2e (Qwen3.5-0.8B), native 6bpw + one-time lm_head dequant, MergedColumn/QKV + TP loaders | **WIP**: open mul1-codebook inf-scale bug (fp32 MLP activations, fold-mul1-to-dense workarounds in flight), decode-only maturity |

## What we have that they don't

- **Recipe-replication serving stack**: exllama dense dispatch for GPTQ,
  BLOCK_KN_SIZE 256, int8-per-token-head KV + **fd_rdna2 flash-decode
  including batched multi-query q≤4 for MTP verify** (their FA explicitly
  punts spec-verify), softmax-segments (recipe 0003) port.
- **MTP machinery**: GDN spec-conv anchoring (correctness fix), kpack
  small-M GEMV (the 10× regression fix), fp16-dot prefill, unquantized-MTP
  checkpoint detection, MoE+MTP load fix, 35B MTP benchmarking.
- **AWQ W4 kernels**: HIP kpack GEMV + `moe_awq_gemm_rdna2` (their tree has
  no AWQ path), the W4 KPACK small-M/fp16-dot fixes.
- Validation corpus: verify/ harness (validate/decode-rate/soak), BM35
  results, NEXT_STEPS diagnosis archive (prefix-cache anomaly, AR verdict,
  power-limit regression).

## Ranked adoption candidates (for a future session)

1. **GDN prefill HIP kernels** — fixes Triton GDN JIT hangs (we hit these:
   the 16k c=8 hang class) and attacks our AWQ 41k prefill pain (~12 min
   eager fp16-GDN prefill). Port: 5 csrc files (largely portable) +
   dispatch integration into our `qwen_gdn_linear_attn.py` (adapt to our
   tree). Effort: ~half a day incl. parity tests (theirs are in
   `tests/kernels/quantization/test_gdn_prefill_*_rdna2.py`).
2. **GPTQ W4A16 MoE + prefill kernels** — closes the 35B GPTQ gap
   (−16–21% vs AWQ native) and gives GPTQ dense prefill configs. Effort:
   ~half a day; benefits from doing #1 first (same dispatch area).
3. **Compile-collectives fix + stock-custom-AR gate** — the
   `use_custom_op_collectives=True` recipe (tiny) applies directly to our
   tree; then a single TP2 re-test of `VLLM_FORCE_CUSTOM_ALL_REDUCE`
   (their +30–34% was TP=4; our plugin-based rejection tested different
   code). Effort: ~1–2 h.
4. **DSV4 family stack** (MXFP4/W8A8/sparse-MLA/indexer) — enables a model
   family we can't serve; their integration is partially deferred, so
   expect wiring work. Effort: 1–2 days. Only if DSV4 becomes a target.
5. **EXL3** — wait for their mul1 fix to land; evaluate then.

## Where things live in their tree (quick pointers)

- GDN prefill: `csrc/rocm/gdn_prefill_*.cu`, callers at
  `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:94-150,1485,1591`
- FA-RDNA2: `csrc/rocm/fa_rdna2.cu`, `vllm/v1/attention/ops/fa_rdna2_backend.py`,
  `vllm/v1/attention/backends/rdna_attn.py`, gate `VLLM_USE_RDNA2_FA`
- W4A16: `csrc/rocm/{q_gemm_rdna2,q_gemm_rdna2_prefill,moe_q_gemm_rdna2}.cu`,
  `vllm/model_executor/kernels/linear/mixed_precision/rdna2_w4a16.py`
- MXFP4: `csrc/rocm/mxfp4_dot2_*.cu`, `vllm/model_executor/layers/fused_moe/experts/rdna2_mxfp4_moe.py`
- DSV4: `csrc/rocm/{gemm_w8a8_fp8_dense,moe_w8a16*,sparse_mla,indexer_paged_mqa}_rdna2.cu`,
  `vllm/models/deepseek_v4/amd/rocm.py`
- Custom AR: `vllm/distributed/device_communicators/custom_all_reduce.py:179-186`,
  `vllm/platforms/rocm.py:1007-1010` (env `VLLM_FORCE_CUSTOM_ALL_REDUCE`)
- Compile-collectives fix: commit 70e65f933, `vllm/platforms/rocm.py:1105-1118`
- EXL3: `csrc/rocm/exl3_*.cu`, `vllm/model_executor/layers/quantization/exl3.py`
