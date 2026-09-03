#!/bin/bash
# Serve btbtyler09/Qwen3.8-27B-GPTQ-4bit on 2x Radeon PRO V620 (gfx1030) —
# the vllm-rdna2-recipe configuration, validated on this fork 2026-08-22.
#
# Measured (idle host, verify/decode-rate.py, greedy):
#   MTP=0: 34.7 t/s @3.5k / 34.1 @13k / 29.9 @39k ctx
#   MTP=2: 41.3 t/s @3.5k / 36.6 @11k / 28.5 @37k ctx (8/8 greedy-identical)
#   MTP=2 refresh 09-03: 34.4 @3.2k / 36.1 @13.4k / 29.5 @37k (acceptance
#   2.04 vs 2.35 — per-step basis within 4% of the 08-22 numbers); c=8
#   guidellm 59.6 out t/s. If numbers sag: CHECK CARD POWER LIMITS FIRST
#   (Item 8b — a lowered cap cost ~40% and rocm-smi lied about it; use
#   amd-smi metric --showpower under load).
# vs the fp16-KV AWQ stack's 25.7 / 14.2 / 6.3.
#
# Key ingredients (each individually reversible):
# - --linear-backend exllama: GPTQ W4 through the Exllama kernel. DO NOT let
#   the RDNA2 W4 kernels auto-select for GPTQ on this tree — they produced
#   garbage output (exclamation tokens) on this checkpoint.
# - BLOCK_KN_SIZE 256 in csrc q_gemm.cu (recipe patch 0004): +4.5%.
# - int8_per_token_head KV + FD_RDNA2=1 (flash-decode plugin): flat context
#   slope; the batched multi-query kernel handles MTP verify (q<=4) in one
#   KV pass.
# - MTP=2 via the checkpoint's own head (qwen3_5_mtp), greedy-lossless.
# - FULL_DECODE_ONLY graphs: startup graph capture 13 s (vs 434 s with
#   FULL_AND_PIECEWISE at 131k ctx).
#
# Cards: production defaults to 2,3 (0,1 are the benchmark pair). First
# start after code changes cold-compiles ~10 min; subsequent starts are fast
# (compile cache + 13 s capture). Send 2-3 warmup requests before judging.

export VLLM_RDNA2_NATIVE_GDN=0
export VLLM_RDNA2_NATIVE_AWQ=1
export VLLM_RDNA2_FP16_GEMV=1
export VLLM_RDNA2_NATIVE_PAGED_ATTN=0
export VLLM_ENGINE_READY_TIMEOUT_S=1200

# Recipe runtime bundle
export FD_RDNA2=1
export FD_CHUNK=512
export FD_TILE=32
export FD_WARPS=8
export FD_MAXQ=4
# AR_RDNA2 push-based allreduce: VALIDATED 2026-09-03 and REJECTED on this
# stack — correct (8/8 greedy identical, 11-min soak clean) but SLOWER:
# 21.4-21.9 vs 24.9-25.0 t/s single-stream decode (-12%), neutral at c=8
# (33.2 vs 33.4). The recipe's +1.9% did not reproduce here (their stack:
# torch 2.11, plain MTP-less serving; ours: MTP k=2 multiplies the
# per-call dispatch overhead). To re-test: needs ROCM_HOME=/opt/rocm AND
# a /opt/rocm/bin/hipcc symlink to /usr/bin/hipcc for the extension to
# build (Gentoo layout — see NEXT_STEPS Item 8).
export AR_RDNA2="${AR_RDNA2:-0}"
export GPU_MAX_HW_QUEUES=4
export HIP_FORCE_DEV_KERNARG=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TORCH_BLAS_PREFER_HIPBLASLT=0
export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
export VLLM_ROCM_USE_AITER=0
export VLLM_ROCM_USE_AITER_MOE=0

# Pin to a fixed card pair (0,1 are the benchmark pair)
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-2,3}"

MTP="${MTP:-2}"
SPEC_ARG=""
[ "${MTP}" != "0" ] && SPEC_ARG="--speculative-config {\"method\":\"qwen3_5_mtp\",\"num_speculative_tokens\":${MTP}}"

exec /mnt/Dev/vllm-rdna2/venv/bin/vllm serve btbtyler09/Qwen3.8-27B-GPTQ-4bit \
    --dtype float16 \
    --kv-cache-dtype int8_per_token_head \
    --linear-backend exllama \
    --tensor-parallel-size 2 \
    --max-model-len 131072 \
    --max-num-seqs 8 \
    --max-num-batched-tokens 8192 \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    ${SPEC_ARG} \
    --compilation-config '{"mode":3,"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --language-model-only \
    --skip-mm-profiling \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3 \
    --gpu-memory-utilization 0.90 \
    --served-model-name Qwen3.8-27B-GPTQ \
    "$@"
