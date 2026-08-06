#!/mnt/Dev/vllm-rdna2/venv/bin/python
# SPDX-License-Identifier: Apache-2.0
"""RDNA2 decode profiler for the AWQ path (GPU kernel timing on ROCm).

Why this structure:
  - On ROCm 7.2 the new torch.profiler(CUDA activity) captures NO GPU kernels,
    but the legacy torch.autograd.profiler(use_cuda=True) DOES (different path).
  - The legacy profiler only sees the calling process, so we force the v1 engine
    to run IN-PROCESS via VLLM_ENABLE_V1_MULTIPROCESSING=0 and TP=1. This loses
    the all-reduce signal (no TP) but captures every per-rank compute kernel
    (AWQ gemm, GDN attention, unquantized gemm, etc.) with real GPU time.

env knobs: BENCH_TP (default 1), BENCH_MAX_MODEL_LEN, BENCH_MAX_TOKENS, BENCH_PROMPT.
"""

import os

# Must be set before importing vllm so the engine runs in the main process.
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

import logging  # noqa: E402

# Silence a scary-but-benign model-info cache log (caught; doesn't block load).
logging.getLogger("vllm.model_executor.models.registry").setLevel(logging.CRITICAL)

import torch  # noqa: E402

import vllm  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402

MODEL = os.environ.get("BENCH_MODEL", "QuantTrio/Qwen3.6-27B-AWQ")
TP = int(os.environ.get("BENCH_TP", "1"))  # keep 1 for in-process GPU profiling
MAX_MODEL_LEN = int(os.environ.get("BENCH_MAX_MODEL_LEN", "8192"))
MAX_TOKENS = int(os.environ.get("BENCH_MAX_TOKENS", "64"))

PROMPT = os.environ.get("BENCH_PROMPT", "Explain gradient descent in three sentences.")
WARMUP = os.environ.get("BENCH_WARMUP", "warmup")


def build_llm() -> LLM:
    return LLM(
        model=MODEL,
        dtype="float16",
        tensor_parallel_size=TP,
        max_model_len=MAX_MODEL_LEN,
        enforce_eager=True,  # clean per-op attribution
        speculative_config={"method": "mtp", "num_speculative_tokens": 2},
        language_model_only=True,
    )


def main() -> None:
    print(f"vllm={vllm.__version__}  file={vllm.__file__}")
    print(f"torch={torch.__version__}  gpu={torch.cuda.get_device_name(0)}")
    print(f"TP={TP}  max_model_len={MAX_MODEL_LEN}  (in-process profiling)")

    llm = build_llm()
    sp = SamplingParams(max_tokens=MAX_TOKENS, temperature=0)

    # Warmup (excluded): fill caches, JIT triton, etc.
    llm.generate([WARMUP], sp)
    torch.cuda.synchronize()

    print("\n>>> starting profiled generate ...")
    # Legacy profiler: the one that actually captures GPU kernel time on ROCm.
    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        llm.generate([PROMPT], sp)
        torch.cuda.synchronize()
    print(">>> done\n")

    sep = "=" * 110
    print(sep)
    print("TOP OPS BY GPU (CUDA) TIME")
    print(sep)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=35))

    # Attribute the copy/reshape churn to its top-level (non-aten) caller so we
    # know whether to cut it in GDN, AWQ, attention, etc.
    conversion_ops = {
        "aten::to",
        "aten::copy_",
        "aten::_to_copy",
        "aten::reshape",
        "aten::contiguous",
        "aten::clone",
        "aten::slice",
    }

    def top_level_parent(ev):
        p = ev.cpu_parent
        while p is not None and (
            p.name.startswith("aten::") or p.name.startswith("prim::")
        ):
            p = p.cpu_parent
        return p.name if p is not None else "<root>"

    by_parent: dict[str, float] = {}
    by_parent_count: dict[str, int] = {}
    for ev in prof.function_events:
        if ev.name in conversion_ops:
            parent = top_level_parent(ev)
            # self_cuda_time_total is in microseconds
            t = getattr(ev, "self_cuda_time_total", 0.0) or 0.0
            by_parent[parent] = by_parent.get(parent, 0.0) + t
            by_parent_count[parent] = by_parent_count.get(parent, 0) + 1

    print(sep)
    print("COPY/RESHAPE CHURN BY TOP-LEVEL CALLER (self CUDA us)")
    print(sep)
    print(f"{'us total':>12} {'%':>7} {'count':>9}  caller")
    print("-" * 110)
    total = sum(by_parent.values()) or 1.0
    for parent, us in sorted(by_parent.items(), key=lambda kv: -kv[1])[:20]:
        print(
            f"{us:12.0f} {100 * us / total:6.1f} {by_parent_count[parent]:9d}  {parent}"
        )


if __name__ == "__main__":
    main()
