# Thatch-cloud Qwen3.8 playbook comparison

## Source

Reviewed
[`Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B`](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B)
at commit `95589fb2ebdd7aa400e1daa89dcb3a8d457756bc` (2026-10-03).
License: repository glue MIT; derived patches retain upstream terms.

This note marks each item as already applied here, worth trying, or not
applicable. It is not a port and does not copy code.

## Result boundary

Thatch reports **offline complete-runtime** measurements on two P150A cards:
128.65 TG / 3,320 PP at a 4,096-token context (repeat 127.74 TG / 3,344 PP),
using a T16/DSpark five-component experimental recipe. Those are not streaming
serving numbers; current-recipe concurrent users and sustained generation are
unqualified there.

This repository's admitted numbers come from the serialized P150x2 service:

| Metric | onix admitted | Thatch report |
| --- | --- | --- |
| B=1 decode | 23.56 tok/s, 64 generated tokens | 128.65 TG at 4K, offline complete runtime |
| B=8 fixed-width batch | 174.65 tok/s aggregate, 21.83 per user | Not reported |
| continuous mixed trace | 36.8–39.9 tok/s delivered | Not reported |
| prefix cache | 0.409x TTFT / 0.574x wall on hits | 1.81x cached prefill at 4K, serving disabled |
| speculative decode | MTP implemented, off by default | DSpark/T16 experimental recipe |

The 128.65 TG and 23.56 tok/s numbers are not directly comparable: different
workload, measurement boundary, speculative stack, and serving state. The
correct next comparison is the new `qwen38-benchmark` on the same prompt,
generation length, and mesh, with both services in the same cache and
speculation mode.

## Playbook and gotchas mapping

| Thatch item | Local status |
| --- | --- |
| Record cards, firmware, fabric, runtime, weights, tokenizer and workload before tuning | Already applied by the receipt and authority model, but the new benchmark should record the serving flags and cache mode more explicitly. |
| Separate cold load, prefill, request setup and warm decode | Already applied in the qwen38 receipt tools and the new benchmark's TTFT/wall fields; the `x_onix_decode_evidence` path carries stage timings. |
| Use complete-generation TG, not draft or kernel throughput | Already applied in the MTP receipt contract. |
| Balanced A/B/B/A ordering and repeated roots | Already applied in the B=8 and prefix-cache admissions. |
| Device numbers can change after reset; resolve `by-id` after the reset | **Worth trying.** Local profiles pin physical IDs 0/1 after an owner reset; a serial-keyed check would make the assignment robust to re-enumeration. |
| Assert logical and physical mesh degree agree before loading weights | **Worth trying.** The service requires the exact mesh and both devices, but does not obviously compare fabric degree telemetry. |
| Descriptor channel count is asserted; choose from measured links | **Worth trying.** The P150x2 profile uses the two-device P300 alias; four-link and two-link descriptors differ. |
| Two Blackhole devices are a P300 to the model stack | Already documented and handled by the P150x2/P300 build. |
| `l1_small_size=24576` is required for `ttnn.conv1d` | Already applied in the service device open and the vLLM preflight settings. |
| vLLM Qwen GDN needs `--no-enable-prefix-caching` | Already recorded in the vLLM dependency profile; not yet a packaged serving lane. |
| Batch limits come from head count, not just memory | Already reflected in the fixed B=8 admission and its non-claim; arbitrary wider batching is not admitted. |
| Demo defaults assume four cards and `shard` sampling | Already handled by the P150x2 host-sampling patch; on-device sampling is not certified at TP2. |
| `QWEN_SDPA_BF8` helps at B=8 long context but is a precision trade at B=1 | Already recorded in `authority.ncl`: long-context KV requires it, with a bounded scope. |
| Watchdog outside the device op; reset after a hang | Already applied in the MTP watchdog and service timeout path. |
| `tt-smi` absent from non-interactive PATH | **Worth trying.** The owner helper resolves `tt-smi` explicitly; any new timer should do the same. |

## Upstream fixes

| Upstream item | Local relevance |
| --- | --- |
| [tt-metal PR 53320](https://github.com/tenstorrent/tt-metal/pull/53320) GDN FIR batch-bound correction | **Worth checking.** No local patch names this fix. Re-check before the next tt-metal bump; do not assume the local `gdn-*` patches include it. |
| [tt-metal issue 54724](https://github.com/tenstorrent/tt-metal/issues/54724) B32 matmul divisibility at TP2 | **Not applicable to the served B=8 profile.** Relevant if B16/B32 is revisited. |
| [tt-metal issue 54725](https://github.com/tenstorrent/tt-metal/issues/54725) short-prompt batched prefill core limit | **Worth checking.** The local grouped-prefill eligibility core is independent; compare the same `B * value_heads / TP` bound before widening. |

## ttsim CI replay

Thatch runs a simulator suite on a dedicated AMD runner with no devices, no
network, 16 CPU equivalents and 64 GiB RAM, serialized with hardware work. Its
`ttsim-ci.md` says the simulator workflow starts on an experiment branch.

This repository already has Nix checks (`ttsim-runtime-bundle`,
`ttsim-single-chip-smoke`, `ttsim-gdn-reference-smoke`,
`ttsim-gdn-input-validation`) but no GitHub workflow directory. A replay check
modeled on Thatch would need:

1. a pinned simulator asset and a device-free runner;
2. a representative Qwen decode/prefill trace that the simulator actually
   models;
3. an external wall-clock timeout and a recorded blocked result when the
   simulator cannot run the path.

A check is now added: `ttsim-qwen-gdn-replay` runs the GDN recurrence at the
Qwen3.8-27B TP=2 shape (24 value heads per shard, head_dim 128, three steps).
The current simulator records a **blocked** receipt for
`ALU_ACC_CTRL_Zero_Flag_disabled_src=1 (keep SrcB denormals) not modeled`
instead of silently skipping. A different error fails the check. It is wired
into the required package-check catalog.

This is one representative Qwen primitive, not a full-model replay. The
existing GDN smoke remains the small-shape device-free gate.

## Actions taken

- The repeatable `qwen38-benchmark` now provides the same-corpus comparison the
  issue asks for; see `docs/qwen38-benchmark.md`.
- The remaining items above are recorded, not silently claimed.
