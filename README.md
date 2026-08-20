<!-- markdownlint-disable MD001 MD041 -->
<!-- ---------------------------------------------------------------------------
     A17T RDNA2 fork README prefix
---------------------------------------------------------------------------- -->

# A17T vLLM RDNA2 fork (`main+rdna2`)

vLLM ported and tuned for **AMD RDNA2 (gfx1030, Radeon PRO V620)**, maintained
by A17T. This branch is based on upstream `main` (post-v0.26.1rc0) and carries
the RDNA2 enablement, native kernels, and decode optimizations below.

## What we did

1. **RDNA2 platform bring-up**: gfx10x detection, MPLinear/kernel choosers,
   6 native HIP kernels (AWQ/GPTQ W4A16 GEMM, AWQ MoE, GDN decode, paged
   attention), fast-startup/compile fixes, TurboQuant FP8 guards.
2. **Decode optimizations** (measured on `cyankiwi/Qwen3.8-27B-AWQ-INT4`,
   compressed-tensors W4 group_size=32, TP2, single-request, short context):
   - rdna2 fp16 GEMV for unquantized M=1 projections (9% -> ~30%+ MBU)
   - **W4A16 M=1 GEMV** over the packed-int4 layout with per-group affine
     dequant, dispatched via an opaque custom op (`vllm.rdna2_w4a16_linear`)
     so the M==1 branch executes with the real runtime M at CUDA-graph
     capture (dynamic-shape guards are not re-evaluated at runtime)
   - deterministic launch heuristics after exhaustive on-device sweeps
     (Triton's do_bench L2-flush picks measurably worse configs)
3. **gfx906-mobydick port** (adapted from
   `ai-infos/vllm-gfx906-mobydick`): MTP attention correctness fixes
   (query_mask_1 padding rows, tile-loop bounds, 3D split-softmax for
   multi-token decode), Torch SDPA fallback paths, NCCL/graph-capture
   stability workarounds, FP8 emulation (fp16 activations + bitwise E4M3
     decode — no `tl.float8` in IR, which crashes Triton on gfx1030), GDN
   kernel launch tuning.

## Results

| date | config | tok/s (decode, batch 1) |
|---|---|---|
| 2026-08-17 | baseline (your command, original tree) | 2.2 |
| 2026-08-17 | + W4A16 GEMV + opaque dispatch + kernel gates | 14.5 |
| 2026-08-18 | + `VLLM_RDNA2_FP16_GEMV=1` | 16.2 |
| 2026-08-18 | + GEMV v2 (BN=32/BK=64 dual-group heuristic) | 19.1 |
| 2026-08-19 | + HIP K-packed W4A16 GEMV (streaming loads, split-K) | **25.2** |

Effective kernel bandwidth of the int4 GEMV went 11-24 GB/s (native kernel at
gs=32) to **171-229 GB/s**. Weight-bandwidth roofline for this setup is
~55-65 tok/s; the remaining gap is the int4 GEMV instruction cost, attention,
and framework overhead. Earlier Qwen3.6-27B-AWQ (AutoAWQ gs=128) work also
produced 9 -> 24 tok/s via the fp16 GEMV alone (see `RDNA2_SUMMARY.md`).

## Open items

- **GDN recurrent-state bugs (Phase 4)**: (a) the native GDN / paged-attn
  kernels emit a correct first token then garbage on Qwen3.8; (b) MTP
  speculative decoding accepts drafts (mean acceptance ~1.9 at k=2) but
  corrupts output after the first accepted tokens — the target-side GDN state
  at verify time is the suspect. Both families block enabling the natives and
  MTP for serving; both are documented in `RDNA2_SUMMARY.md`.
- Long-context decode (paged attention) and batch scaling after that.

## Usage

See `serve_q38_27b.sh` (validated serve config) and `RDNA2_SUMMARY.md`
(full investigation notes, env table, build/test workflow). Key env toggles:
`VLLM_RDNA2_FP16_GEMV`, `VLLM_RDNA2_NATIVE_GDN`, `VLLM_RDNA2_NATIVE_AWQ`,
`VLLM_RDNA2_NATIVE_PAGED_ATTN`, `VLLM_RDNA2_TRITON_ATTN_TUNING`,
`VLLM_TRITON_ATTN_NUM_PAR_SOFTMAX_SEGMENTS`, `VLLM_TORCH_SDPA_*`.

First startup after code changes takes ~10 min (fresh AOT compile + Triton
autotune); the cache is warm per config afterwards.

<!-- ---------------------------------------------------------------------------
     Upstream vLLM README follows
---------------------------------------------------------------------------- -->

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-light.png" width=55%>
  </picture>
</p>

<h3 align="center">
Easy, fast, and cheap LLM serving for everyone
</h3>

<p align="center">
| <a href="https://docs.vllm.ai"><b>Documentation</b></a> | <a href="https://blog.vllm.ai/"><b>Blog</b></a> | <a href="https://arxiv.org/abs/2309.06180"><b>Paper</b></a> | <a href="https://x.com/vllm_project"><b>Twitter/X</b></a> | <a href="https://discuss.vllm.ai"><b>User Forum</b></a> | <a href="https://slack.vllm.ai"><b>Developer Slack</b></a> |
</p>

🔥 We have built a vLLM website to help you get started with vLLM. Please visit [vllm.ai](https://vllm.ai) to learn more.
For events, please visit [vllm.ai/events](https://vllm.ai/events) to join us.

---

## About

vLLM is a fast and easy-to-use library for LLM inference and serving.

Originally developed in the [Sky Computing Lab](https://sky.cs.berkeley.edu) at UC Berkeley, vLLM has grown into one of the most active open-source AI projects built and maintained by a diverse community of many dozens of academic institutions and companies from over 2000 contributors.

vLLM is fast with:

- State-of-the-art serving throughput
- Efficient management of attention key and value memory with [**PagedAttention**](https://blog.vllm.ai/2023/06/20/vllm.html)
- Continuous batching of incoming requests, chunked prefill, prefix caching
- Fast and flexible model execution with piecewise and full CUDA/HIP graphs
- Quantization: FP8, MXFP8/MXFP4, NVFP4, INT8, INT4, GPTQ/AWQ, GGUF, compressed-tensors, ModelOpt, TorchAO, and [more](https://docs.vllm.ai/en/latest/features/quantization/index.html)
- Optimized attention kernels including FlashAttention, FlashInfer, TRTLLM-GEN, FlashMLA, and Triton
- Optimized GEMM/MoE kernels for various precisions using CUTLASS, TRTLLM-GEN, CuTeDSL
- Speculative decoding including n-gram, suffix, EAGLE, DFlash
- Automatic kernel generation and graph-level transformations using torch.compile
- Disaggregated prefill, decode, and encode

vLLM is flexible and easy to use with:

- Seamless integration with popular Hugging Face models
- High-throughput serving with various decoding algorithms, including *parallel sampling*, *beam search*, and more
- Tensor, pipeline, data, expert, and context parallelism for distributed inference
- Streaming outputs
- Generation of structured outputs using xgrammar or guidance
- Tool calling and reasoning parsers
- OpenAI-compatible API server, plus Anthropic Messages API and gRPC support
- Efficient multi-LoRA support for dense and MoE layers
- Support for NVIDIA GPUs, AMD GPUs, Intel GPUs, and x86/ARM/PowerPC CPUs. Additionally, diverse hardware plugins such as Google TPUs, Intel Gaudi, IBM Spyre, Huawei Ascend, Rebellions NPU, Apple Silicon, MetaX GPU, and more.

vLLM seamlessly supports 200+ model architectures on Hugging Face, including:

- Decoder-only LLMs (e.g., Llama, Qwen, Gemma)
- Mixture-of-Expert LLMs (e.g., Mixtral, DeepSeek-V3, Qwen-MoE, GPT-OSS)
- Hybrid attention and state-space models (e.g., Mamba, Qwen3.5)
- Multi-modal models (e.g., LLaVA, Qwen-VL, Pixtral)
- Embedding and retrieval models (e.g., E5-Mistral, GTE, ColBERT)
- Reward and classification models (e.g., Qwen-Math)

Find the full list of supported models [here](https://docs.vllm.ai/en/latest/models/supported_models.html).

## Getting Started

Install vLLM with [`uv`](https://docs.astral.sh/uv/) (recommended) or `pip`:

```bash
uv pip install vllm
```

Or [build from source](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/index.html#build-wheel-from-source) for development.

Visit our [documentation](https://docs.vllm.ai/en/latest/) to learn more.

- [Installation](https://docs.vllm.ai/en/latest/getting_started/installation.html)
- [Quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)
- [List of Supported Models](https://docs.vllm.ai/en/latest/models/supported_models.html)

## Contributing

We welcome and value any contributions and collaborations.
Please check out [Contributing to vLLM](https://docs.vllm.ai/en/latest/contributing/index.html) for how to get involved.

## Citation

If you use vLLM for your research, please cite our [paper](https://arxiv.org/abs/2309.06180):

```bibtex
@inproceedings{kwon2023efficient,
  title={Efficient Memory Management for Large Language Model Serving with PagedAttention},
  author={Woosuk Kwon and Zhuohan Li and Siyuan Zhuang and Ying Sheng and Lianmin Zheng and Cody Hao Yu and Joseph E. Gonzalez and Hao Zhang and Ion Stoica},
  booktitle={Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles},
  year={2023}
}
```

## Contact Us

<!-- --8<-- [start:contact-us] -->
- For technical questions and feature requests, please use GitHub [Issues](https://github.com/vllm-project/vllm/issues)
- For discussing with fellow users, please use the [vLLM Forum](https://discuss.vllm.ai)
- For coordinating contributions and development, please use [Slack](https://slack.vllm.ai)
- For security disclosures, please use GitHub's [Security Advisories](https://github.com/vllm-project/vllm/security/advisories) feature
- For collaborations and partnerships, please contact us at [collaboration@vllm.ai](mailto:collaboration@vllm.ai)
<!-- --8<-- [end:contact-us] -->

## Media Kit

- If you wish to use vLLM's logo, please refer to [our media kit repo](https://github.com/vllm-project/media-kit)
