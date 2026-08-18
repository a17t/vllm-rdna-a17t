#!/bin/bash
# Serve cyankiwi/Qwen3.8-27B-AWQ-INT4 on 2x Radeon PRO V620 (gfx1030) with
# the current RDNA2 tree. Measured 2026-08-18: ~19.1 tok/s single-request
# decode at short context (round-1: 2.2 -> 14.5 with the W4A16 GEMV;
# round-2: +FP16_GEMV=1 and GEMV v2 dual-group tuning).
#
# Env notes:
# - NATIVE_GDN / NATIVE_PAGED_ATTN stay off: the native GDN and paged-attn
#   kernels still produce garbage after the first token on this checkpoint
#   (open Phase-4 bug).
# - NATIVE_AWQ has no effect on this checkpoint anymore: its gs=32 layers
#   are auto-routed to the Triton W4A16 kernel + M=1 GEMV fast path.
# - FP16_GEMV=1: Triton GEMV for the unquantized GDN projections + lm_head
#   (validated, +1.7 tok/s).
# - MTP speculative decoding is NOT enabled: acceptance is healthy (~1.9 at
#   k=2) but the GDN spec-state machinery corrupts output after the first
#   accepted tokens (open bug, see RDNA2_SUMMARY). Do not enable until fixed.
#
# Startup: first launch after code changes takes ~10 min (fresh AOT compile
# + Triton autotune); the cache is warm for this exact config afterwards
# (purge ~/.cache/vllm/torch_compile_cache if you touch traced code).

export VLLM_RDNA2_NATIVE_GDN=0
export VLLM_RDNA2_NATIVE_AWQ=1
export VLLM_RDNA2_FP16_GEMV=1
export VLLM_RDNA2_NATIVE_PAGED_ATTN=0
export VLLM_ENGINE_READY_TIMEOUT_S=1200
exec /mnt/Dev/vllm-rdna2/venv/bin/vllm serve cyankiwi/Qwen3.8-27B-AWQ-INT4 \
    --dtype float16 \
    --kv-cache-dtype float16 \
    --tensor-parallel-size 2 \
    --max-model-len 262144 \
    --reasoning-parser qwen3 \
    --compilation-config '{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --gpu-memory-utilization 0.98 \
    -dp 2
