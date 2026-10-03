# Cloudflare Clef on Tenstorrent

Cloudflare/Clef (27B) and Cloudflare/Clef-Flash (9B) now serve the released
`POST /v1/systemone` decision contract on the local TT-Metal 0.77 candidate.
These are **separate services** from the [Qwen3.8 autoregressive structured
reads](qwen38-structured-reads.md): the Qwen3.5 backbone and vision tower run
on TT-NN, while Cloudflare's released joint schema head and output embedding
run on the host. No generation or token-by-token decode is involved. The
checkpoints and released head implementation are pinned by content hash; see
`pkgs/qwen36/clef-models.nix` and `pkgs/qwen36/default.nix`. The released code
and checkpoint license files are Apache-2.0.

## Prepare the device and checkpoint

- Clef requires an exclusive `1x2` mesh on **physical P150a PCIe IDs 0 and 1**.
  Flash opens **ID 0 only**, including on a host that also has ID 1. The
  wrappers set `MESH_DEVICE=P150x2` and `P150`, respectively; neither falls back
  to another device or simulator. Stop any existing device owner first.
- The pinned runtime is TT-Metal 0.77; upstream Blackhole requirements are
  firmware **>= 19.8.1** and TT-KMD **>= 2.8.0**. Check the actual machine before
  use. The bounded run below used firmware 19.11.0.0 and TT-KMD 2.10.0.
- Put the read-only model in the Nix store and the TT tensor cache on a
  **spacious, writable filesystem**. The cache directory basename must equal
  the checkpoint revision and must not be inside the checkpoint. Do not share
  a cache between revisions or variants. The checkpoints fetch 12 and 4
  safetensors shards respectively; they are not pulled in by the ordinary
  `qwen38` build.

```sh
clef=$(nix build --no-link --print-out-paths .#clef-checkpoint)
flash=$(nix build --no-link --print-out-paths .#clef-flash-checkpoint)
cache_root=${XDG_CACHE_HOME:-$HOME/.cache}/tt-clef
mkdir -p "$cache_root/2f3de3dd85f379784083b0814d997ab627200f0c" \
  "$cache_root/17f0b0ad64efb65d273590632833508766b2aae6"

# Run ONE of these while its physical devices are free; startup loads the full backbone.
nix run .#clef-serve -- --model-path "$clef" \
  --cache-path "$cache_root/2f3de3dd85f379784083b0814d997ab627200f0c" \
  --port 8190
nix run .#clef-flash-serve -- --model-path "$flash" \
  --cache-path "$cache_root/17f0b0ad64efb65d273590632833508766b2aae6" \
  --port 8189
```

If `$HOME` lacks space, first place `cache_root` above on a spacious writable
filesystem. Separately, give TT-Metal 0.77's **compiled binary** cache its own
private directory outside the revision-specific TT tensor caches passed as
`--cache-path`:

```sh
kernel_cache=/path/on/spacious/filesystem/clef-tt-metal-kernels
install -d -m 700 "$kernel_cache"
```

Prefix **either** `nix run` command above with `TT_METAL_CACHE="$kernel_cache"`.
Missing kernels compile there on first request, after `stage:ready`; restarts
reuse matching binaries by build key. The private-cache physical run below
verified this opt-in override; it does not change the wrappers' defaults or
imply a warm-request speedup.

By default the service binds `127.0.0.1` and has no authentication: do not
expose it directly to untrusted clients. It serializes requests because GDN,
KV, and vision state are mutable. `--max-length` defaults to 2,048 input tokens
and accepts 1–16,384; the upper bound is not a validated hardware capacity.
`HF_HUB_OFFLINE=1` is set by both wrappers. Startup rejects wrong variants,
revisions, head hashes, architecture, device IDs, and unwritable or incorrectly
named cache directories. Stop with SIGINT/SIGTERM to close the TT mesh.

## Request contract

A request has the matching `model`, `state`, and nonempty `questions` map.
Question `type` is `noul`, `choice`, or `score`. For choice, `criteria` maps
option IDs to descriptions; score uses a list of descriptions. The response
contains `model`, per-question `answers`, and `usage` (`output_tokens: 0`).
This is the released Clef joint-head contract, **not** the iterative
Qwen3.8 structured-read extensions (`steps`, `think`, `samples`, etc.).

```sh
curl -sS http://127.0.0.1:8190/v1/systemone \
  -H 'content-type: application/json' \
  -d '{"model":"Cloudflare/Clef","state":"The sky is blue.","questions":{"color":{"type":"choice","instructions":"What color is the sky?","criteria":{"blue":"blue sky","green":"green sky"}}}}'
```

On the bounded Clef run this returned `color.choice: "blue"`, confidence
`0.9973`, probabilities `blue: 0.9973`, `green: 0.0027`, and
`usage.input_tokens: 132`. Use `Cloudflare/Clef-Flash` and port 8189 for the
Flash service. JSON images are base64 **image data URLs** in `images`; videos
are lists of frame lists in `videos`, with each frame an image data URL.
The processor also accepts `media_kwargs` as a JSON object. A record may have
images **or** videos, but the current TT vision splice does not support both
in one record; that request returns HTTP 400 instead of silently dropping
media. Question or input encoding errors return HTTP 400; inference failures
return HTTP 500. A wrong model name returns HTTP 400, not a cross-checkpoint
answer. These wrappers serve the pinned releases, not arbitrary HF checkpoints.

## Bounded evidence and limits

[Physical smoke receipt](evidence/clef-physical-smoke-2026-10-03.json) records
the exact checkpoint revisions, package outputs, mesh placement, request
outcomes, repeated-request state reset, and a 1,342-token (>1,024) prompt on
each variant that traversed a recurrent-state chunk boundary. Text choices,
noul/score, real PNG images, and four-frame video requests were exercised.
The package recorded there (after an import-order and malformed-metadata
error-handling cleanup) reproduced the text-choice responses on both physical
meshes. Both smoke services were stopped. This is **physical silicon**, not
TTsim: the simulator cannot model the GDN `ALU_ACC_CTRL_Zero_Flag_disabled_src=1` path
([0.77 port](tt-metal-0.77-port.md#exact-gdn-blocker)). It does not establish
a model-wide accuracy score, performance target, production readiness, or
equivalence to a CPU reference over a corpus. The prior Qwen production
service was isolated for the probe and was not restored by it.

## Warm Clef-Flash prefill

Clef-Flash pads each hidden-state prefill segment to the next 128-token GDN
subchunk, rather than always rounding to 128/256/512/1,024. For a 342-token
request, this cuts the work from 512 to 384 rows; for a 1,342-token request,
the second segment falls from 512 to 384 rows. The Clef 27B two-device path
retains the old bucket policy: a paired test showed no TP latency improvement.

On physical P150a ID 0, eight alternating, **warmed** baseline/candidate HTTP
requests per shape had median end-to-end times of 0.521/0.322 seconds (342
tokens) and 1.194/0.987 seconds (1,342 tokens). The rebuilt package's five
warm requests per shape had medians of 0.312 and 1.005 seconds. Recorded choices
and probabilities matched across the paired comparisons. The rebuilt package
also served 640/768/896-row buckets, red PNG image and video requests, and
repeated requests after different inputs. Clef 27B's prior text decisions were
reproduced on the rebuilt two-device package.

The additional Flash buckets **compile on first use**. In this bounded run,
first requests to newly used 640/768/896-row buckets took about 16 seconds;
the first 384-row request took 52 seconds in the freshly rebuilt process,
including other unwarmed kernels. The warm-latency gains do not imply better
cold-start latency or model-wide accuracy. See the [bucket optimization
receipt](evidence/clef-flash-prefill-buckets-2026-10-03.json) for inputs,
paired samples, package identities, response checks, and limits. Both
verification services were stopped; the prior Qwen owner remains isolated.

## Horner inverse output reuse

The GDN prefill inverse reuses its local Horner result tensor as the output
of each `ttnn.add` after the previous `ttnn.matmul` has finished reading it.
This removes 30 newly allocated add outputs per 32-column inverse without
changing the arithmetic order, matmul outputs, or inter-request state. On
physical TT-Metal 0.77, full FP32 inverse outputs for real Clef and
Clef-Flash block batches, and the tested full BF16 hidden states, were
byte-identical to the original solver.

For Clef 27B on P150x2, warmed, in-process alternating HTTP requests had
median baseline/reuse times of **3.269/3.125 seconds** (1,342 tokens; eight
samples per arm) and **1.664/1.584 seconds** (182 tokens; six per arm).
Each paired cycle improved on the 27B baseline. The separately rebuilt
package served choice, noul, score, long-prompt and repeated-state requests
on both physical meshes, with the recorded decisions and probabilities
unchanged. The Clef-Flash 1,342-token paired medians were 1.012/1.018
seconds: **no Flash long-request speedup is demonstrated**.

See the [inverse output-reuse receipt](evidence/clef-horner-output-reuse-2026-10-03.json)
for allocation counts, inverse/hidden tensor shapes and equality, raw 27B
samples, package identities and limits. Newly rebuilt processes compiled
kernels on their first requests; neither cold-start speedup nor corpus-wide
numerical equivalence is established. All test meshes were closed afterward.

## Warm-cache process startup

On physical TT-Metal 0.77 with the existing checkpoint, TT tensor, and HOME
kernel caches retained, the dominant Clef 27B P150x2 startup phase was
constructing the 64 language layers, not checkpoint discovery, mesh opening,
the CPU head, or visual initialization. In an instrumented baseline run,
`tp_common.shard_w` occupied **103.36 seconds across 320 calls** within model
initialization: it converted and transposed full CPU checkpoint weights before
`ttnn.as_tensor` discarded that input on a tensor-cache hit. Profiled child and
parent phases overlap and instrumentation changes absolute times.

The tensor-parallel shard helper now uses TTNN's public `load_tensor` on an
existing dtype/layout-matched cache file, before copying the CPU weight.
TTNN 0.77's own `as_tensor` cache-hit path loads that same flatbuffer onto
the same mesh; a missing or unreadable cache retains the original
checkpoint-backed `as_tensor` creation/repair path. This follows upstream's
lazy-weight pattern. The Flash single-device loader is unchanged. No kernels
are prewarmed or KV caches allocated before readiness.

Two independent **one-process-per-arm** unprofiled Clef 27B startup pairs,
both using the same package Python/TTNN environment, warmed checkpoint/tensor
caches, physical P150x2, and production mesh-descriptor isolation, measured
process start to `stage:ready`: **132.548 → 51.360 seconds** with the existing
default HOME kernel cache, and **165.538 → 96.982 seconds** when both arms
used a new private `TT_METAL_CACHE` on spacious `/datapool`. These are
bounded reductions of 61% and 41%, respectively, not a stable latency
distribution; system conditions differed between pairs. Both packages gave
exactly equal parsed response objects on the private-cache arm for nine
requests apiece: first-use and warmed 132-token text choice, 1,339-token
cross-chunk choice, real red PNG image choice, four-frame video noul, and
changed-state followed by restored-state checks. A separate TP pair passed
three joint noul/ordinal-score requests per arm, including changed and
restored states with unchanged probabilities and usage. First-use kernels
compiled **after** readiness; warm-request latency is not claimed to improve.

Clef-Flash 9B on P150 has no tensor-parallel shard-helper calls. Its
private-cache startup pair was **70.043 → 72.596 seconds** and all nine
corresponding text/long/image/video/changed-state response objects matched
exactly; no Flash startup improvement is demonstrated. An earlier default
HOME-cache pair measured 75.921 → 70.082 seconds, but the rebuilt Flash long
request then returned HTTP 500 because TTNN could not write its XIP ELF:
`No space left on device` under `$HOME/.cache/tt-metal-cache` (the root
filesystem reported zero available bytes at that time). Repeating both arms
with the **same supported private kernel-cache override** passed the long,
image, and video requests, including both shapes warmed. No HOME cache or
system files were altered to bypass the environment failure.

In a further patched TP instrumented run, `from_pretrained` took 30.463
seconds, the 64 language-layer constructors 23.581 seconds, and `shard_w`
11.170 seconds across the same 320 calls; MLP loading occupied 7.104 seconds
and GDN loading 10.056 seconds inside those layer phases. The vision
initialization took 6.402 seconds. KV allocation took 0.418 seconds on the
**first HTTP request**, not during startup. The next measured startup
hotpath is remaining language-layer weight setup and cached TTNN device
loads, not request-side KV/kernel work to move ahead of readiness. See the
[bounded startup receipt](evidence/clef-startup-cache-hit-2026-10-03.json)
for package identities, individual paired timings, response checks, profiler
scope, preserved log paths, and resource limits.
