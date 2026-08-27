#!/bin/bash
# Serve Qwen3.6-35B-A3B (MoE, 256 experts / 8 active) on 4x Radeon PRO V620
# (gfx1030). Benchmark round 2026-08-22/23 (BM35-RESULTS.md, guidellm,
# 256-in/100-out, 60 s arms):
#
#   GPTQ (btbtyler09) TP4 EP0 MTP0: 44.6 t/s @c1 / 396 @c64   (throughput opt)
#   GPTQ TP4 EP0 MTP2:              51.5 @c1 (+15%)           (single-stream)
#   AWQ  (QuantTrio)  TP4 MTP0:     50.7 @c1 / 491 @c64       (fastest quant)
#   AWQ  TP4 MTP2:                  64.2 @c1 (+27%)           (best single)
#
# - EP (--enable-expert-parallel) loses at every concurrency on this PCIe
#   host at DP=1 — keep it off.
# - TP2 loses to TP4 decisively — keep TP4.
# - MTP acceptance is only ~1.1 on this model family (BF16 drafter vs
#   quantized target divergence on MoE routing), so MTP wins single-stream
#   but LOSES at concurrency >= 8. Set MTP=2 for interactive/single-user,
#   MTP=0 for throughput serving.
# - AWQ quant uses the native moe_awq_gemm_rdna2 HIP kernel (16-21% faster
#   than the GPTQ Triton expert path); default below is AWQ. Override with
#   MODEL=btbtyler09/Qwen3.6-35B-A3B-GPTQ-4bit (adds --linear-backend
#   exllama automatically? no — pass EXTRA_ARGS='--linear-backend exllama').

export VLLM_RDNA2_NATIVE_GDN=0
export VLLM_RDNA2_NATIVE_AWQ=1
export VLLM_RDNA2_FP16_GEMV=1
export VLLM_RDNA2_NATIVE_PAGED_ATTN=0
export VLLM_ENGINE_READY_TIMEOUT_S=1200
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800

export FD_RDNA2=1
export FD_CHUNK=512
export FD_TILE=32
export FD_WARPS=8
export FD_MAXQ=4
export AR_RDNA2="${AR_RDNA2:-0}"
export GPU_MAX_HW_QUEUES=4
export HIP_FORCE_DEV_KERNARG=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TORCH_BLAS_PREFER_HIPBLASLT=0
export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
export VLLM_ROCM_USE_AITER=0
export VLLM_ROCM_USE_AITER_MOE=0

# Benchmark pair is 0,1; production takes all four for TP4.
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0,1,2,3}"

MODEL="${MODEL:-QuantTrio/Qwen3.6-35B-A3B-AWQ}"
MTP="${MTP:-0}"
SPEC_ARG=""
[ "${MTP}" != "0" ] && SPEC_ARG="--speculative-config {\"method\":\"qwen3_5_mtp\",\"num_speculative_tokens\":${MTP}}"

exec /mnt/Dev/vllm-rdna2/venv/bin/vllm serve "${MODEL}" \
    --dtype float16 \
    --kv-cache-dtype int8_per_token_head \
    --tensor-parallel-size 4 \
    --max-model-len 131072 \
    --max-num-seqs 64 \
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
    --served-model-name Qwen3.6-35B-A3B \
    "$@"
