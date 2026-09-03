#!/bin/bash
# bm35_config.sh — one benchmark config for the Qwen3.6-35B-A3B round.
#
# Usage: bm35_config.sh <label> <TP> <EP:0|1> <MTP:0|2> [model] [extra-args...]
#
# Starts the server on port 8000 (all TP cards via HIP_VISIBLE_DEVICES=0-3),
# waits for health, warms up, runs verify/validate.py (save or compare), then
# guidellm at c in {1,8,32,64} (60 s each). Results appended to
# /tmp/opencode/gptq35b/results.md and logs in /tmp/opencode/gptq35b/<label>/.

set -u

LABEL="$1"; TP="$2"; EP="$3"; MTP="$4"; MODEL="${5:-btbtyler09/Qwen3.6-35B-A3B-GPTQ-4bit}"
shift 4 || true
[ $# -gt 0 ] && shift || true

WORK=/tmp/opencode/gptq35b/${LABEL}
mkdir -p "${WORK}"

BASE_VAL=/tmp/opencode/gptq35b/validate_baseline.json

# --- server ---------------------------------------------------------------
EP_FLAG=""
[ "${EP}" = "1" ] && EP_FLAG="--enable-expert-parallel"
SPEC_ARG=""
[ "${MTP}" != "0" ] && SPEC_ARG="--speculative-config {\"method\":\"qwen3_5_mtp\",\"num_speculative_tokens\":${MTP}}"

export HIP_VISIBLE_DEVICES=0,1,2,3 CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_RDNA2_NATIVE_GDN=0 VLLM_RDNA2_NATIVE_AWQ=1 VLLM_RDNA2_FP16_GEMV=1
export VLLM_RDNA2_NATIVE_PAGED_ATTN=0
export VLLM_ENGINE_READY_TIMEOUT_S=1200 VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800
export FD_RDNA2=1 FD_CHUNK=512 FD_TILE=32 FD_WARPS=8 FD_MAXQ=4 AR_RDNA2=0
export GPU_MAX_HW_QUEUES=4 HIP_FORCE_DEV_KERNARG=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TORCH_BLAS_PREFER_HIPBLASLT=0 FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
export VLLM_ROCM_USE_AITER=0 VLLM_ROCM_USE_AITER_MOE=0

LINEAR_FLAG="--linear-backend exllama"
case "${MODEL}" in
  *AWQ*) LINEAR_FLAG="" ;;
esac

setsid nohup /mnt/Dev/vllm-rdna2/venv/bin/vllm serve "${MODEL}" \
  --dtype float16 --kv-cache-dtype int8_per_token_head \
  ${LINEAR_FLAG} \
  --tensor-parallel-size "${TP}" ${EP_FLAG} \
  --max-model-len 131072 --max-num-seqs 64 \
  --max-num-batched-tokens 8192 --enable-chunked-prefill --enable-prefix-caching \
  ${SPEC_ARG} \
  --compilation-config '{"mode":3,"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --language-model-only --skip-mm-profiling \
  --gpu-memory-utilization 0.90 --port 8000 --served-model-name bm35 \
  "$@" > "${WORK}/server.log" 2>&1 < /dev/null &
SERVER_PID=$!

# --- wait for health (up to 30 min: fresh spec-decode compile hashes can
# take ~20 min on TP4) ---------------------------------------------
READY=0
for i in $(seq 1 180); do
  if curl -sf http://127.0.0.1:8000/health > /dev/null 2>&1; then READY=1; break; fi
  sleep 10
done
if [ "${READY}" != "1" ]; then
  echo "[${LABEL}] SERVER FAILED; tail:" | tee -a /tmp/opencode/gptq35b/results.md
  tail -20 "${WORK}/server.log" | tee -a /tmp/opencode/gptq35b/results.md
  # Kill the whole detached server tree: SERVER_PID is the setsid wrapper,
  # so negative-PID kill the group AND sweep by cmdline.
  kill -9 -- -"${SERVER_PID}" 2>/dev/null
  kill -9 "${SERVER_PID}" 2>/dev/null
  pkill -9 -f 'port 8000' 2>/dev/null
  pkill -9 -f 'VLLM::EngineCore' 2>/dev/null
  pkill -9 -f 'VLLM::Worker' 2>/dev/null
  sleep 3
  exit 1
fi
echo "[${LABEL}] READY after ~$((i*10))s"

# --- warmup (2 shape-matched) + greedy gate -------------------------------
/mnt/Dev/vllm-rdna2/venv/bin/python - > "${WORK}/warm.log" 2>&1 <<'PY'
import json, urllib.request, time
for _ in range(2):
    body = json.dumps({"model":"bm35","prompt":"List the first 25 prime numbers, comma-separated, in order: 2, 3, 5, 7","max_tokens":80,"temperature":0.0}).encode()
    t0=time.perf_counter()
    r = urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8000/v1/completions",
        data=body, headers={"Content-Type":"application/json"}), timeout=600)
    j = json.load(r)
    print(f"warm: {j['usage']['completion_tokens']} tok in {time.perf_counter()-t0:.1f}s :: {j['choices'][0]['text'][:60]!r}")
PY
grep '^warm:' "${WORK}/warm.log"

if [ ! -f "${BASE_VAL}" ]; then
  /mnt/Dev/vllm-rdna2/venv/bin/python /mnt/Dev/vllm-rdna2/vllm-upstream/verify/validate.py \
    --url http://127.0.0.1:8000/v1/chat/completions --model bm35 \
    --save "${BASE_VAL}" > "${WORK}/validate.log" 2>&1
  GATE="baseline-saved"
else
  /mnt/Dev/vllm-rdna2/venv/bin/python /mnt/Dev/vllm-rdna2/vllm-upstream/verify/validate.py \
    --url http://127.0.0.1:8000/v1/chat/completions --model bm35 \
    --compare "${BASE_VAL}" > "${WORK}/validate.log" 2>&1
  GATE=$(grep -oE 'identical=[0-9]+  diverged=[0-9]+' "${WORK}/validate.log" | tail -1)
fi
echo "[${LABEL}] gate: ${GATE}" | tee -a /tmp/opencode/gptq35b/results.md

# --- guidellm sweep --------------------------------------------------------
for C in 1 8 32 64; do
  /mnt/Dev/vllm-rdna2/venv/bin/guidellm run \
    --backend kind=openai_http,target=http://localhost:8000 \
    --profile kind=throughput,max_concurrency=${C} \
    --constraint kind=max_duration,seconds=60 \
    --tokenizer kind=huggingface_auto,model="${MODEL}" \
    --data kind=synthetic_text,prompt_tokens=256,output_tokens=100 \
    > "${WORK}/guidellm_c${C}.log" 2>&1
  # extract headline stats
  STATS=$(/mnt/Dev/vllm-rdna2/venv/bin/python - <<PY
import re
txt = open("${WORK}/guidellm_c${C}.log").read()
# guidellm prints a benchmark table; grab the rate lines
rows = re.findall(r'\|[^\n]*\|', txt)
for r in rows:
    if any(k in r for k in ("Completed", "Request", "Output", "Token", "Total", "Throughput")):
        print(r.strip())
PY
)
  echo "" >> /tmp/opencode/gptq35b/results.md
  echo "## ${LABEL} c=${C}" >> /tmp/opencode/gptq35b/results.md
  echo '```' >> /tmp/opencode/gptq35b/results.md
  echo "${STATS}" >> /tmp/opencode/gptq35b/results.md
  echo '```' >> /tmp/opencode/gptq35b/results.md
  echo "[${LABEL}] c=${C} done"
done

# --- stop server -----------------------------------------------------------
kill "${SERVER_PID}" 2>/dev/null
sleep 6
kill -9 -- -"${SERVER_PID}" 2>/dev/null
pkill -9 -f 'port 8000' 2>/dev/null
pkill -9 -f 'VLLM::EngineCore' 2>/dev/null
pkill -9 -f 'VLLM::Worker' 2>/dev/null
sleep 3
echo "[${LABEL}] COMPLETE" | tee -a /tmp/opencode/gptq35b/results.md
