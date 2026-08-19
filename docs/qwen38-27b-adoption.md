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

The PR's 27B configurations require a P150x4 or P150x8 mesh. This repository's
hardware evidence currently covers P150x1 / P150x2, so Qwen3.8-27B cannot be
placed on the documented training mesh. Deployment to a 4- or 8-card box is:

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

## Upstream state watch

The PR is closed-unmerged. If upstream merges it or the qwen36 path moves,
first re-run `qwen36-check` and the authority export; only update the pin with
new evidence.
