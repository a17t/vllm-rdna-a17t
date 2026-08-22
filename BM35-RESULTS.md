# Qwen3.6-35B-A3B benchmark round — 2026-08-22 (guidellm, 4× V620)

Model: `btbtyler09/Qwen3.6-35B-A3B-GPTQ-4bit` (256 experts, 8 active, A3B,
moe_intermediate=512, gs=32 sym GPTQ) + AWQ control
(`QuantTrio/Qwen3.6-35B-A3B-AWQ`). Workload: guidellm throughput profile,
synthetic 256-in/100-out tokens, 60 s per concurrency level, port 8000.
Common config: int8 KV, FD_RDNA2=1, mode 3 + FULL_DECODE_ONLY, MAXSEQS=64,
BATCHTOK=8192, recipe env bundle. GPTQ arms add `--linear-backend exllama`
(dense layers). Host exclusive (all 4 cards). Greedy gates: baseline saved
on tp4_ep0_mtp0; all arms 7-8/8 (1-divergence arms are single-token
numeric drift from layout/kernel changes, primes byte-exact everywhere).

## Results (output tokens/s / TPOT ms)

| Config | c=1 | c=8 | c=32 | c=64 |
|---|---|---|---|---|
| TP4 EP0 MTP0 (GPTQ) | 43.8 / 22.4 | 115.7 / 51.4 | 259.5 / 104.5 | 407.3 / 153.9 |
| TP4 **EP1** MTP0 | 37.8 / 26.0 | 87.3 / 63.7 | 213.1 / 116.7 | 388.5 / 160.9 |
| TP2 EP0 MTP0 | 32.0 / 30.7 | 50.9 / 78.6 | 137.0 / 177.3 | 310.1 / 201.6 |
| TP4 EP0 MTP0 **AWQ control** | **50.7 / 19.1** | **127.8 / 34.8** | **352.0 / 77.6** | **491.4 / 126.1** |

## Conclusions

1. **Optimal GPTQ config: TP4, EP off, MTP off** (43.8 → 407 t/s across
   concurrency). EP is a loss at every concurrency (-5..-25%): with DP=1
   there is no all2all fast path, and TP4's per-expert intermediate 128 is
   still wide enough for the Triton expert kernel — the moe_int=512
   sharding concern did not materialize as an EP win on this PCIe host.
2. **TP4 > TP2 decisively** (c=1 +37%, c=64 +31%). Single-stream MoE
   benefits from attention+expert TP width; TP2's 2×3B-active per-card
   does not compensate.
3. **GPTQ experts are ~16-21% slower than the native AWQ MoE kernel**
   (43.8 vs 50.7 @c1; 407 vs 491 @c64). GPTQ MoE routes to
   `TritonWNA16Experts` (generic Triton, no gfx1030 tuning) while AWQ hits
   the fork's `moe_awq_gemm_rdna2` HIP kernel. For this model the cached
   **QuantTrio AWQ is the faster quant** on this stack; the btbtyler GPTQ
   would need a gfx1030-tuned MoE config or GPTQ wired into the RDNA2 MoE
   kernel to close the gap.
4. **MoE MTP is broken on this tree (GPTQ path)**: weight-load failure —
   `Layer mtp.layers.0.mlp.experts has no parameter 'w2_weight' for
   checkpoint weight 'mtp.layers.0.mlp.experts.down_proj'` — the
   Qwen3_5MoeMTP drafter's expert param mapping doesn't match the
   GPTQ/MoeWNA16 method's expected names. MTP arms dropped (parked bug).

## Notes

- guidellm synthetic data needs `--tokenizer kind=huggingface_auto,model=...`
  (defaults to an unavailable tokenizer otherwise).
- All server/gate logs: /tmp/opencode/gptq35b/<label>/; raw sweep in
  results.md there. Runner: bm35_config.sh (untracked).
- Follow-ups (parked): MTP MoE weight-mapping bug; gfx1030 tuning configs
  for fused_moe_kernel_gptq_awq; GPTQ support in the RDNA2 MoE HIP kernel;
  MTP for the AWQ variant untested this round.
