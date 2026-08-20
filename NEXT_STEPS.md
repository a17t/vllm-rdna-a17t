# NEXT STEPS — parked work items (updated 2026-08-20)

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

## Item 1b (NEW, ACTIVE): MTP throughput regression — correct-but-slow

Correct MTP runs at 2.5 tok/s (~1.1 s/step) vs corrupted-but-fast 25 tok/s
(81 ms/step) pre-fix. Evidence gathered:
- py-spy: 86% of worker time blocked in
  `num_accepted_tokens_event.synchronize()` (gpu_model_runner.py:2177)
  + `torch.accelerator.synchronize` (:1219) — GPU queue is the long pole
- GPU shows 99% busy (amd-smi) — possibly NCCL spin (TORCH_NCCL_BLOCKING_WAIT=1
  from our rocm.py workaround) rather than real work
- BOTH the reference Triton conv and the anchored native conv are equally
  slow -> NOT kernel cost; slowdown correlates with CORRECT spec semantics
  (acceptance 2.8 vs 2.1), i.e. something downstream of correct anchoring
- conv kernel itself: 4.2us/call captured, 76us eager — exonerated
- non-spec control on same build: 25.16 tok/s — exonerated globally
Next: profile the GPU queue composition during a step (rocprof attach or
in-server event timing around drafter proposal); check whether the draft
forward's eager NCCL collectives serialize with the graph replays; test
with TORCH_NCCL_BLOCKING_WAIT unset; consider capping mamba spec blocks.
MTP stays OFF in serve_q38_27b.sh until this lands.

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
