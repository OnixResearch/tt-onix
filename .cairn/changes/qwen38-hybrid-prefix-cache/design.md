## Context

A Qwen3.8 prefix spans both attention families. The September report derives 72 MiB of raw recurrent state per chip, before convolution and KV. At 2048 tokens, raw BF16 KV adds 64 MiB per chip.

## Decisions

Start with one approved shared system prefix and a bounded in-process cache. Do not persist device addresses or snapshots across restarts.

The core owns a content-bound key, completeness checks, memory accounting, tenant quotas, and eviction order. The shell supplies authenticated tenant scope and measured allocation facts. The TTNN adapter owns snapshot and restore effects.

The key includes exact tokens and all identity inputs from the evidence contract. Similar text is not a match. The entry lists recurrent state, convolution carry, full-attention KV, positions, and any final hidden row required by the continuation route.

Entries are immutable. A continuation receives isolated state through copies or a proven copy-on-write contract. Partial KV blocks require separate ownership before writes. No cached prefix contains live MTP speculation state in the first version.

Stable destination buffers belong to the current trace plan. A restore cannot replace trace-bound allocations. A digest proves content identity, not authorization or correct state.

## Opt-in shared system/tool root candidate

The deployed `prefix-cache-policy.json` remains schema v1: an eligible request
saves its longest complete 1,024-token chunk that fits the KV budget. The
separate `prefix-cache-root-policy.json` is schema v2 and enables a minimum
2,048-token shared root, four preallocated carried-state buffers, and at most
1,024 cached KV blocks. Its physical admission below covers only the named
workloads; this opt-in profile is not deployed.

To exercise it, pass the packaged
`share/qwen36/prefix-cache-root-policy.json` to the existing
`--prefix-cache-policy` option of a candidate service. Do not replace the v1
service policy without a separate physical admission.

On an eligible chat request, render the system/tool header through the same Qwen3.8
template using an empty probe user. Strip only the known final user-message suffix,
then require the header's token IDs to match the actual full prompt up to the saved
chunk boundary. A difference after that boundary may still safely reuse the identical
earlier chunks; changes within the saved chunks, model/template identity, or tenant
cannot. A missing system and tools, a short header, unknown template framing, a
tokenizer mismatch, or a full-prompt boundary uses ordinary leaf caching. The exact
prefix digest still binds the model, package, precision, tenant, and token chunks;
raw system text is not a cache key.

The first eligible cold request saves the root rather than its private last-turn
prefix: the masked prefill trace permits one state save point. Later requests resume
the root's KV blocks and GDN state and may save longer conversation prefixes into
other preallocated buffers. A root is never superseded in place by a descendant; its
blocks are shared read-only and subsequent writes use the request's own KV run. At
most one root is soft-retained per worker. Unpinned leaves evict before the root,
but a live request can evict any unpinned root for KV. A different system/tool header
falls back to ordinary leaf caching while an unrelated root is resident. Structured
reads invalidate unpinned entries; a device failure drops every root and leaf.

Candidate evidence must compare cold and candidate on a common long system/tool
header with more interleaved chats than saved states, including leaf eviction,
cross-chat root hits, exact generated-token parity, memory bounds, and TTFT/wall
latency. The existing v1 physical receipt does not establish a v2 benefit. The
TT-Metal 0.77 simulator does not establish fused-GDN numerical recurrence; do not
promote this candidate from a tokenizer/device-free check alone.

The candidate passed `nix build .#checks.x86_64-linux.qwen36 --no-link -L
--option substituters ''`, including the v1 and v2 policy, token-boundary, KV/state
restore, eviction, isolation, and invalidation probes. A device-free smoke using the
installed Qwen3.8 tokenizer and the real scheduler served six different prompts with
one 3,072-token system root: the first planned `(hit=0, save=3072)`, the next five
planned `(hit=3072, save=4096)`, and the number of entries remained bounded by four.
The installed tokenizer also recognized long system headers with and without tools
and rejected a changed token. This does not exercise silicon, fused-GDN state
numerics, latency, or physical placement.

### Declared shared-root physical screen

`prefix-cache-root-admission-policy.ncl` fixes one workload of five conversations
and three turns each, exceeding the four saved-state buffers. All conversations
use the same long system header; their user passages and history differ. Run
four fresh processes in cold, cached, cached, cold order on the same package,
runtime, model, and physical devices 0 and 1. The first cached request must
save a root of at least 2,048 aligned tokens; all other first turns must hit
that root. Conversation 1 must save a longer leaf on turn 0, then resume from
the root on turn 1 after eviction. The two cached processes must agree on root
length, and every cold and cached completion must decode identical token IDs.

The preregistered limits are cached/cold median TTFT ≤ 0.70 and wall ≤ 0.85
for hits, median miss wall ≤ 1.10 and worst miss wall ≤ 1.25, and startup
device-DRAM drop ≤ 536,870,912 bytes. Compare per-turn ABBA means to cancel
linear host-load drift; retain pair ratios and CPU pressure as diagnostics.
Use a 20-minute timeout for each bounded diagnostic process. This physical
screen cannot clear the separate TT-Metal 0.77 fused-GDN simulator numerical
blocker or by itself authorize production deployment.

### Shared-root hardware screen (2026-09-29)

The declared ABBA screen **admitted the opt-in root profile for this workload
only**. The four fresh processes on britton-desktop used physical PCIe devices
0 and 1, package
`/nix/store/fz8pq66iwji7hnssy1xnqfzw0c037pzr-qwen38-0.1.0`, execution
identity `/nix/store/kw7idl8jhl9vbrrbfbk269y3bzilnkfy-qwen36-execution-identity-0.1.0`,
and TT-Metal `/nix/store/7hplkynh047dsk98cwqpfsaaq6lwyy0m-tt-metal-0.77.0`.
The cached processes used `prefix-cache-root-policy.json` (BLAKE3
`d917d9eaa8ee6961c9f0929b679f68fa7ac4e15a9dd30347aaef77381a4c6e44`);
the others used no prefix-cache policy. Each had the production runtime knobs,
the same model, and a fresh service on port 8011. The diagnostic shell's
`CI=true` was overridden with `CI=false` to match the production unit:
upstream benchmarking utilities import CI-only `infra` when `CI=true`.

The screen ran `scripts/qwen38-prefix-cache-screen.py --shared-system
--conversations 5 --turns 3` with one 9,000-character system header and
distinct user passages. The first-turn prompts were 4,260–4,270 tokens.
The filler is the concatenation, for `i` from 0 through 2399, of
`f"Record {i:05d}: The amber river measured station {i % 71:02d} at dawn; analyst {i % 113:03d} compared the telescope reading with archived notes {i * 37 % 100003:05d}.\n"`;
its SHA-256 is `4ec6c3587595e2d3c981a0af72ae0e40b2ac4a6b87da2df8256d4f6b469e8665`
and the complete workload BLAKE3 is
`a4b2f77b40138e067d7bcef1c510a5f1a6ddf94629f886439f8371dead0bd718`.

All 15 turn keys produced identical token-ID sequences in all four roots.
In both cached processes, the donor saved the 2,048-token root, the other
14 turns resumed it, conversation 1 saved a 4,096-token leaf on its first
turn, and its next turn fell back to the root after eviction. The receipt
measured median hit TTFT ratio 0.6815 (bound 0.70), hit wall ratio
0.8023 (bound 0.85), miss median and maximum wall ratio 0.9339 (bounds
1.10/1.25), and startup allocator-view DRAM drop 364,904,448 bytes (bound
536,870,912). Both adjacent-pair hit-wall medians were 0.801–0.804; CPU
pressure and per-pair ratios remain diagnostics, not burst-resilient gates.
Policy BLAKE3:
`8a3d73ecc890cce9150165e3545ba247679348c825e2aed8add162e6e299fd65`.
The four uncompressed roots and the input-bound receipt are
`pkgs/qwen36/evidence/2026-09-29/shared-root-{a1,b1,b2,a2}.json` and
`shared-root-admission-receipt.json` in the same directory.

This admission does not cover tool-bearing headers, other prompt shapes,
cross-tenant sharing, full-model numerical correctness, or production
rollout. The TTsim fused-GDN numerical blocker above remains. The deployed
v1 policy is unchanged; the production owner service was restored after
the screen.

### Declared shared-tool isolation screen

The separate `prefix-cache-tool-admission-policy.ncl` adds a schema-v4
changed-tool gate to the same five-conversation, three-turn ABBA screen and
the same latency, exact-parity, and 512 MiB startup DRAM limits. Each request
offers the same `lookup_observation(station: integer)` function through the
Qwen3.8 chat template. After the 15 interleaved turns in each fresh process,
the driver sends a fresh chat with the same long system instructions but a
different **description** for that function, not a different function name.
The change must miss the retained cache entirely; its generated token IDs
must match across the two cached and two cold processes. The extra request
does not contribute to the 15-turn timing ratios.

Installed-tokenizer preflight on the candidate package with the explicit
text-only instruction rendered a 4,538-token first prompt and a 2,048-token
system/tool root. Changing only the tool description produced a 4,542-token
prompt that diverged at token 40, inside the saved root. The first cold
attempt, before that instruction, did not return a text-only answer at
conversation 2 turn 0 and was discarded. The driver now instructs the model
to answer from the passage without calling the offered tool and refuses
any tool-call output
rather than treating empty text history as a valid continuation. Four fresh
processes on physical P150x2 devices 0 and 1, bounded to 20 minutes each,
are required. Passing this screen would cover only the declared tool header
and changed-schema probe; the GDN simulator blocker and deployment boundary
remain unchanged.

The 2026-09-29 physical ABBA run admitted the opt-in tool policy. Evidence:
[`tool-root-admission-receipt.json`](../../../pkgs/qwen36/evidence/2026-09-29/tool-root-admission-receipt.json)
and its adjacent `tool-root-a1/b1/b2/a2.json` roots. Four fresh P150x2
processes reported physical PCIe device IDs 0 and 1, TT-Metal 0.77 runtime
`7hplkynh047dsk98cwqpfsaaq6lwyy0m` and candidate package
`i64cdsnjqdvv1vx69c36cndff17ncgns`. The two cached roots resumed 2,048
tokens on 14/15 turns each, including the five interleaved conversations;
all 60 generated turn-token sequences matched the cold roots. The changed
description with the **same function name** resumed zero tokens in both
cached roots, and the four changed-probe outputs matched exactly.

Admitted measurements: resumed-turn median client TTFT ratio 0.6838
(limit 0.70), median wall ratio 0.8036 (limit 0.85), miss wall ratio
1.0037 (limit 1.10 median and 1.25 maximum), startup allocator DRAM
drop 364,904,448 bytes (limit 536,870,912). This covers an *offered* tool
with text-only answers, not real function invocation or broader tool
schemas. The runtime policy remains opt-in and production still uses its
prior v1 policy; TTsim's fused-GDN numerical gate is still blocked.

The serialized production owner was restored: its systemd unit is active,
`/healthz` reports `ready: true` on port 8000, and the diagnostic port 8011
has no listener.

## Validation

Positive tests cover an identical prefix, an empty suffix, and repeated continuation under the same route. Negative tests cover a tokenizer change, head replacement, tenant mismatch, missing recurrent state, wrong tensor shape, corrupted digest, partial restore, and exhausted memory.

Matched hardware compares cold prefill with lookup plus full restore. It includes cache misses and eviction. A useful hit must improve complete request latency without a failed memory or quality gate.

The serving host also runs unrelated work, so its load drifts between roots. Admission therefore measures four fresh roots per workload in ABBA order (cold, cached, cached, cold). Each gate compares the per-turn mean of the two cached roots with that of the two cold roots, which cancels linear drift. Per-pair ratios and per-turn host CPU pressure are diagnostics, not gates.

## Alternatives and rollback

A KV-only cache is rejected. Unbounded per-conversation snapshots are rejected. Cross-restart persistence and automatic cross-tenant sharing are deferred.

On a miss, use cold prefill. On failed restoration, reset or quarantine before cold fallback. The feature stays opt-in until the exact profile passes.
