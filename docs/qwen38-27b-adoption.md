# Qwen3.8-27B adoption (tt-metal PR 53543)

This document records the repository adoption of
[tenstorrent/tt-metal pull 53543](https://github.com/tenstorrent/tt-metal/pull/53543),
"Enable Qwen3.8-27B on the existing qwen36 Blackhole path".

## What the PR does

The PR registers `Qwen/Qwen3.8-27B` as an `HF_MODEL` weight-swap on the
main-line `models/demos/blackhole/qwen36` path. It:

- registers the Qwen3.8-27B checkpoint, prefill-chunk sizes, and tokenizer
  fallback in `models/tt_transformers/tt/model_config.py`;
- adds opt-in BF4 weight caches (`QWEN36_LM_HEAD_BF4`, `QWEN36_MLP_DOWN_BF4`,
  `QWEN36_GDN_BF4`) stored under distinct `.bfp4` cache names so default BFP8
  bins are never overwritten;
- keeps `QWEN_SDPA_BF8=1` as the long-context KV path;
- adds the `traced_150k` demo case and clips long prompts before tokenization;
- makes the vLLM adapter import-safe when Qwen3.5 multimodal helpers are
  absent, so text-only `vllm==0.10.1.1` can serve.

The PR targets P150x4 and P150x8 meshes only (4-way / 8-way tensor parallel).
It is **closed but not merged** upstream (`merged: false`).

## Adoption decision

The repository's packaged tt-metal runtime is pinned to `v0.74.0-rc6`, which
does **not** contain the `models/` tree at all. The qwen36 path — and every
change this PR builds on — exists only on tt-metal `main`. A full runtime pin
bump to main would rebuild the C++ stack and risk the validated RWKV-7 and
training deployments. This change therefore adopts the PR in the least
disruptive form:

1. **Pin the adopted source tree** at the PR head commit
   `d102e7f4512060e09e580e40cf26a870bfa88c55`. The tree already contains the
   PR on top of the qwen36 base, so no patch file is required.
2. **Layer the `models/` tree** from that pin over the packaged 0.74 runtime
   as a standalone deployable (`pkgs/qwen36`). No change to the cached tt-metal
   0.74 derivation, no RWKV/training rebuild.
3. **Gate the adoption on device-free evidence** (`qwen36-check`): the adopted
   modules must import against the packaged runtime, the opt-in BF4 gates must
   keep their documented BFP8 defaults, and the vLLM adapter must import with
   vLLM absent.

The runtime-compatibility question (0.74 vs main) is answered by the
device-free check, not by assumption. If the import gate ever fails against
0.74, the checked-in `qwen36` package records the exact blocker and the
bounded alternative (a main runtime pin) becomes the next change.

## Reviewed authority

The adoption facts live in `pkgs/qwen36/authority.ncl` with the contract in
`authority-contract.ncl` and the exported, committed copy in
`generated/authority.json`. The Nix validator (`authority.nix`) admits the
authority only when it matches:

- upstream repository `tenstorrent/tt-metal.git`, branch `qwen38-qb2`,
  commit `d102e7f...`, base `c71ea3ca...`, `merged_upstream = false`;
- PR 53543 facts (117 additions / 40 deletions / 9 files);
- release `Qwen/Qwen3.8-27B`, revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`,
  architecture `Qwen3_5ForConditionalGeneration`, 18 checkpoint shards,
  weights excluded;
- hashed metadata artifacts (license, config, tokenizers, preprocessors,
  checkpoint index) with exact Nix and BLAKE3 identities.

Drift in any recorded fact fails the package evaluation. A separate
`qwen36-authority-negative` check proves a drifted commit is rejected.

## Deployable surface

- `nix build .#qwen36` — the deployable; `qwen36`, `qwen36-check`,
  `qwen36-serve` binaries, plus the installed `models/` tree.
- `qwen36-check` — device-free adoption gate (no hardware).
- `qwen36-serve` — text-only vLLM 0.10 entrypoint; fails closed when vLLM is
  not importable.
- `nix build .#checks.x86_64-linux.qwen36` and
  `.#checks.x86_64-linux.qwen36-authority-negative` — CI-shaped gates.

## Deployment (hardware)

The 27B configurations require a P150x4 or P150x8 mesh for the PR's certified
path. This repository additionally carries a **P150x2 (`P300`)** build so the
two-P150 deployment can attempt Qwen3.8-27B at 2-way tensor parallel without
waiting for a larger box.

On a P150x4/x8 box:

```sh
HF_MODEL=Qwen/Qwen3.8-27B MESH_DEVICE=P150x4 nix run .#qwen36 -- \
  -m pytest models/demos/blackhole/qwen36/demo/text_demo.py -k 'traced_4k and not batched'
```

For long-context and the reported throughput, set the PR flags:

```sh
export QWEN_SDPA_BF8=1                # BF8 SDPA + paged KV (long context)
export QWEN36_LM_HEAD_BF4=1           # opt-in BF4, distinct .bfp4 caches
export QWEN36_MLP_DOWN_BF4=1
export QWEN36_GDN_BF4=1
```

Measured by the PR author on P150x4 (with all four flags): traced-4k
28.69 tok/s, traced-150k 25.54 tok/s, vLLM `max_model_len=4096` 22.63 e2e
tok/s. These are upstream-reported numbers, not repository evidence.

## Non-claims

The adoption makes no silicon, placement, profiler, thermal, or performance
claim on this repository. The device-free check proves imports and gate
behavior only. Physical validation must run on a P150x4/x8 box (or a bounded
ttsim smoke for the single-device 9B path) before any runnable claim.

## Carried P150x2 (P300) build

`pkgs/qwen36/p150x2-tp.patch` adds a TP=2 path on top of the adopted PR. It:

- registers `P150x2 → (1, 2)` in the demo and test mesh maps;
- adds `P300` prefill-chunk entries (`2` in 1024-token units) for
  Qwen3.5/3.6/3.8-27B — `P300` is the unit-test built-in device name for
  two Blackhole devices (`determine_device_name`) and already appears in
  `get_trace_prefill_supported_seq_lens`;
- advertises **host sampling** at TP=2: `supports_sample_on_device` is made
  mesh-conditional because on-device sampling is certified only for 1x4/1x8
  (the Top-K 65536-logits/device shard limit; Qwen3.8-27B gives 124160
  logits/device at TP=2).

Why TP=2 divisibility is safe for this checkpoint: 24 heads, 4 KV heads,
16/48 linear-attention heads, hidden 5120, intermediate 17408, and vocab
248320 all divide by 2; no KV-head replication is needed (that path exists for
TP=8). The carried tree is not silicon-validated: the prefill tuning falls back
to frozen TP=4 values, and decode throughput at TP=2 (host sampling, 2x
per-device weight traffic versus 4-way TP) is unmeasured.

Validation remaining for P150x2:

1. device-free gate — done (`qwen36-check` TP=2/P300 probes PASS);
2. simulator — open a 1x2 mesh under the packaged `libttsim_bh_x2.so`, run
greedy `test_generate_tp.py` Paris, carry GDN recurrent state, receipt marked
`simulated`;
3. hardware — same commands on the two-P150 box, `tests/pcc_thresholds.json`
TP=2 rows, then TTFT/tok/s with the BF4/BF8 flags.

## Hardware evidence (2026-08-19, britton-desktop)

The two-P150 box validated the adopted stack on real silicon with the packaged
0.74 runtime (no runtime bump):

- **Silicon op roundtrip** — device open on physical chip 1, host→device,
BF16 matmul 32x1024x1024, device→host; cosine 0.999968 against torch
  (`evidence/2026-08-19/hw-silicon-matmul.log`).
- **Qwen3.5-9B single-device generation — PASS**: the demo's own paged path
  (`_run_paged_generation`) produced `"The capital of France is Paris."` with
first generated token `Paris` and a non-degenerate continuation
  (`evidence/2026-08-19/hw-9b-paged-generate.log`). Physical chip 1,
`TT_VISIBLE_DEVICES=1`, `TT_MESH_GRAPH_DESC_PATH` unset (the ambient p150_x2
descriptor cannot map onto one device — the same hazard `rwkv7-chat`
documents).

Two blockers were found and fixed on the way, both recorded as carried patches
or package fixes:

1. `generate_tp` crashed on single-device models: `reset_tp()` calls
   `reset_state()` on every layer, but single-device full attention only
   exposes `reset_cache()` — carried as `generate-tp-reset.patch`. The PR's
   own oracle test box was unchecked upstream; the single-device oracle had
   never been run. (`generate_tp`/`prefill_tp` remain TP-only entry points —
   the 4D fractured-residual layout is wrong for single-device use; the 9B
   validation above uses the served paged path instead.)
2. The packaged 0.74 runtime include tree lacks
   `api/compute/experimental/matmul_custom.h` (and more), which the qwen36 SDPA
   kernels JIT-include. The package replaces the overlay's `hw/inc/api`
   subtree with files from the package-matched source.
3. The 0.74 bindings lack `ttnn.transformer.chunk_gated_delta_rule`. The
   `gdn-seq-fallback.patch` probes for that operator and uses
   `gated_delta_attn_seq` when the fused path is absent.
4. The installed runtime lacks required TP JIT kernel sources under
   `ttnn/cpp`. The package overlays the complete package-matched `ttnn/cpp`
   tree, including `all_gather_minimal_matmul_async`.

The TP=2 attempt started at 20:00 EDT. It opened the exact 1x2 mesh and
initialized `FABRIC_1D`. The console showed all 64 layers loaded and P300 cache
generation complete.

The run continued through sequence-path JIT compilation until the host reboot
at 21:46 EDT. The reboot interrupted the run before its two-hour timeout and
removed the temporary log. No fatal error appeared before the reboot, but the
run produced no retained token, PCC, TTFT, or throughput result.

Both inference services are active after the reboot. The TP=2 path remains
unvalidated until a new bounded attempt produces a durable receipt.

## Upstream state watch

The PR is closed-unmerged. If upstream merges it or the qwen36 path moves,
first re-run `qwen36-check` and the authority export; only update the pin with
new evidence.
