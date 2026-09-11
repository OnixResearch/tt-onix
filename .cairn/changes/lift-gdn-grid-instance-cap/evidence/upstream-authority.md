# Upstream source authority and runtime capability

Recorded 2026-09-11 for change `adopt-fused-gdn-decode-verify-op`.

## Task A1: upstream source

### PR 53482 - `ttnn.transformer.decode_gated_delta_rule`

| Field | Value |
| --- | --- |
| Repository | `tenstorrent/tt-metal` |
| State | open, `merged: false`, `draft: false` |
| Head | `ctxbot/tt-metal`, branch `gdn-decode-fused-dw` |
| Head commit | `a7100468ec7a7df7b9c7699719a66779fc52d8d0` |
| Base | `tenstorrent/main` at `3c8d365bf1e8a9458178676cd218981e27d24bbf` |
| Merge commit (unused) | `d77b1d8e1994c9c736f650a7ee83ae87c4dcdb3c` |
| Commits | 11 |
| Diff size | 18 files, `+1569`, `-0` |
| Labels | `pr-priority:p2`, `pr-risk:high`, `pr-complexity:high` |
| Issue comments | 0 |
| Review comments | 0 |
| Requested reviewers | `razorback3`, `dongjin-na`, `dvartaniansTT`, `aczajkowskiTT`, `sdawle-TT` |

Commits named in the PR body: `09eb57476438274f2a061728967ea636efa7b3bd` (op),
`965377373cc` (op hardening), `a8bbcac4b4c` (wrapper).

**Review state:** no human review. Five reviewers were requested and none
responded. This repository must treat the patch as an unreviewed source.

### PR 53587 - `qwen36` P300 traced-demo integration stack

Open and unmerged. It stacks ahead of PR 53482 and carries the traced-demo
integration plus the same op content. Head commits named in its body:
`f9c4dbae318`, `9a28790b64a`, `58e66a8e546`, `c4b82a9706b`, `b9aa2e29a71`,
`687f0e77dd2`, plus `a360a6ea96d` (formatting only).

Its silicon ladder on 2x Blackhole p150a in a 1x2 mesh, Qwen3.8-27B:

| Case | Decode |
| --- | ---: |
| `traced_128` | 19.16 tok/s |
| `traced_4k` | 19.03 tok/s |
| `traced_8k` | 18.84 tok/s |

Independent re-run of `traced_128`: 19.12 tok/s. Unfused baseline on the same
node: 17.75 tok/s.

### Issue 50475 - `fused_recurrent_gated_delta_rule`

A separate, employee-authored effort on branch `atupe/qwen36-mtp-v2`. Reported
on P150x4: decode `0.345` ms per layer against `0.541` for the composite, and
verify `2.3` ms per candidate per layer against about `18` ms before.

This kernel and PR 53482's kernel carry different names and come from different
authors. They are not the same artifact. This repository measured neither.

## Task A2: pinned runtime capability

Inspected runtime:
`/nix/store/dr9bj8qwflylq39zl5ajcx42cdrb1bv9-tt-metal-0.77.0`.

Binding counts in `lib/python3.13/site-packages/ttnn/_ttnn.so`:

| Symbol | Occurrences |
| --- | ---: |
| `chunk_gated_delta_rule` | 3 |
| `gated_delta_rule` | 3 (substring of the row above) |
| `decode_gated_delta_rule` | **0** |
| `fused_recurrent_gated_delta_rule` | **0** |
| `recurrent_gated_delta_rule` | **0** |

`ttnn/operations/transformer.py` in the same runtime binds only
`SDPAProgramConfig`, `PagedCacheGeometryOverride`, and `SparseKVFormat`, plus
golden functions. It binds no gated-delta decode op.

**Conclusion:** the pinned runtime can accept neither fused kernel. Both
candidates require vendoring. The pin does carry `chunk_gated_delta_rule`,
which is the prefill path and is a separate finding.

## Patch applicability

`patch -p1 --dry-run` against the pinned source tree
`/nix/store/5cafvl5cvjzck5hjl36ixj48mpx6j28f-source` (tt-metal `v0.77.0`):

- exit code `0`
- files checked `18`
- hunk failures `0`

The PR was written against `main` at `3c8d365`, which is newer than `v0.77.0`.
The patch nevertheless applies to the pin without a reject.

Vendored copy: `.pi/qwen38-gdn-fused/pr53482.diff` (1738 lines).

## Build implication

Every existing patch in `pkgs/qwen36/` targets the Python `models/` tree. The
package overlays JIT kernel sources and Python modules only. It never rebuilds
the C++ runtime.

The fused op is C++. `ttnn.transformer.decode_gated_delta_rule` requires
recompiling `ttnn_op_transformer` and the `_ttnn` nanobind module. Vendoring it
therefore requires a patched tt-metal derivation, not a package patch.

This is a named cost, not a blocker: a second derivation keeps the shared
`tt-metal` 0.77 unchanged for RWKV-7 and the training platform, and Nix caches
the rebuild after the first build.

## Non-claims

This file records source facts and an applicability result. It contains no
throughput measurement, no parity result, and no admission. The upstream
numbers are upstream claims and are not local evidence.
