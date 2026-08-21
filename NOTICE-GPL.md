# GPL-3.0-or-later vendored components (NOTICE)

The following directories in this fork contain code vendored verbatim from
[`vllm-rdna2-recipe`](https://github.com/leapdragon/vllm-rdna2-recipe)
(Copyright (C) 2026 Aron Hsiao, licensed **GPL-3.0-or-later**):

- `plugins/fd_rdna2/` — flash-decode attention kernel + batched multi-query
  verification path; gated on `FD_RDNA2=1`
- `plugins/ar_rdna2/` — push-based one-shot all-reduce over uncached device
  staging buffers; gated on `AR_RDNA2=1`
- `verify/validate.py`, `verify/decode-rate.py`, `verify/prefill-rate.py`,
  `verify/soak.py` — server-side A/B harnesses for correctness and perf

These components are **optional** (env-gated, no-op unless the env var is set)
and are kept in separately identified directories so that the rest of the fork
remains cleanly licenseable.

## Upstream contribution

**Do not include the contents of `plugins/fd_rdna2/`, `plugins/ar_rdna2/`, or
`verify/` in any upstream contribution or PR against a non-GPL-licensed
project.** The fork's core (`vllm/`, `csrc/`, build files, docs) is intended
to remain Apache-2.0-compatible; only the directories listed above carry GPL
obligations.

If the upstream path ever requires the kernel or all-reduce functionality, a
clean-room re-implementation (or an explicit dual-licensing arrangement with
the original author) is the appropriate route.

## Attribution

> vllm-rdna2-recipe: vLLM on the Radeon PRO V620 (Navi 21 / gfx1030).
> Copyright (C) 2026 Aron Hsiao.
> Licensed under the GNU General Public License v3.0 or later.

Adaptation note: `plugins/fd_rdna2/fd_rdna2/__init__.py` has been modified
for this fork's 0.26.x tree (separate K/V split views + dedicated float32
scale caches instead of the recipe's combined-buffer layout; the fast path
now calls the in-tree Apache-2.0 kernel
`vllm/model_executor/layers/attention/rdna2_fd_decode.py`). The original
combined-buffer kernel is retained unmodified at
`plugins/fd_rdna2/fd_rdna2/fd_kernel2.py` as reference for the future
batched multi-query (MTP verify) port. `plugins/ar_rdna2/` is vendored
unmodified. All of these files remain GPL-3.0-or-later.

See `LICENSE-GPL-3.0.txt` (the GPL text) for the full license under which the
files in `plugins/fd_rdna2/`, `plugins/ar_rdna2/`, and `verify/` are
distributed.
