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

## Opt-in NixOS full-27B service (source only)

The default module also provides `hardware.tenstorrent.clef` for the full
`Cloudflare/Clef` release, **not** the single-card Flash model. Enabling it
requires an explicitly configured, already pinned Qwen rollback unit. It keeps
that unit installed but removes Qwen's `multi-user.target` boot pull, makes
both units conflict, and boots `clef-p150x2.service` instead. Qwen-only
hosts are unchanged. An external mesh sidecar that still `Wants=Qwen` can
re-pull Qwen despite this unit's boot setting: change its backend and route to
Clef in the **same reviewed host deployment**. This module alone is not a
Britton-desktop rollout.

```nix
hardware.tenstorrent = {
  enable = true;
  # Keep the site's existing Qwen modelPath, package, and rollback settings.
  qwen38.enable = true;
  clef = {
    enable = true;
    modelPath = "/home/brittonr/models/cloudflare-clef-2f3de3dd85f379784083b0814d997ab627200f0c";
    # Default: /dev/shm/clef-p150x2/2f3de3dd85f379784083b0814d997ab627200f0c
    # cachePath may instead name an existing private revision directory.
  };
};
```

The module's default tmpfs cache parent and exact revision directory are
created root-owned with mode 0700; custom cache paths must already exist. The
preflight requires the checkpoint's revision marker to match
`2f3de3dd85f379784083b0814d997ab627200f0c` and refuses a missing,
symlinked, unwritable, misnamed, or nonempty TT tensor cache before starting the
model. This ensures a restart does not silently claim the fresh-cache identity
while reusing unattested tensors. A failed conversion leaves partial tensors:
inspect the stopped owner and clear **only that service-owned revision cache**
under separate authorization before a manual retry. No automatic cleanup,
restart loop, Qwen fallback, or unrelated cache sweep is performed. tmpfs is
volatile: every boot needs a fresh conversion and adequate free memory/tmpfs
capacity (the prior full-model cache reached about 24.2 GB). The distinct
persistent kernel cache lives under the unit's private
`/var/lib/clef-p150x2/tt-metal-kernels`; HOME and XDG cache are private too.

Clef uses `Type=simple` because `clef-serve` has no systemd-notify
contract. Systemd's active state is **not** readiness: check
`GET http://127.0.0.1:8190/v1/identity` and `/v1/models` for the
pinned full variant and revision before sending `POST /v1/systemone` or
advertising the model through the mesh. The port is loopback-only. For manual
Qwen rollback, first prevent Clef being re-pulled by the host/mesh dependency
and stop the Clef owner, then start the retained Qwen unit and restore the
Qwen mesh route; merely stopping Clef while mesh still Wants it is unsafe.
No NixOS service switch or TT execution follows from evaluating this module.

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
revisions, checkpoint file hashes, architecture, device IDs, and unwritable or incorrectly
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

## Served identity

Startup checks the whole checkpoint before it opens a TT device. The package
generates `share/qwen36/clef-release.json` from the per-file hashes in
`clef-models.nix`, and both wrappers pass it as `--release-manifest`. The service
hashes every pinned file of the variant with SHA-256: the backbone shards, the
index, the configurations, tokenizer and processor files, the chat template, the
license, the joint head and `joint_schema_model.py`. It refuses to start if a
pinned file is missing or different, or if the directory holds any entry other
than these and the Nix `revision` marker or Hugging Face `README.md`,
`.gitattributes` and `.cache` metadata. The imported `clef_release_model.py`
must equal the pinned `joint_schema_model.py`. This reads the whole checkpoint
once per start, about 55.0 GB for Clef and 19.1 GB for Flash; requests hash
nothing.

The response `model` only echoes the request. `GET /v1/identity` returns the
identity object built after these checks:

| Field | Value |
|---|---|
| `schema` | `clef-tt-identity/1` |
| `model` | public ID, `Cloudflare/Clef` or `Cloudflare/Clef-Flash` |
| `accepted_models` | every request `model` value the service admits |
| `variant` | `clef` or `clef-flash` |
| `hf_repo`, `hf_revision` | the pinned Hugging Face repository and revision |
| `checkpoint_files` | file name to SHA-256 for every verified checkpoint file |
| `checkpoint_sha256` | SHA-256 of `checkpoint_files` as sorted-key, compact UTF-8 JSON |
| `release_code_sha256` | SHA-256 of the imported released encoder and head module |
| `tt_tensor_cache` | `empty-at-startup` or `preexisting-unattested` |
| `max_length`, `max_body_bytes` | input-token bound and request-body limit |
| `endpoint` | `POST /v1/systemone` |

`GET /v1/models` returns an OpenAI-style list with the public ID and the same
object under `identity`, so an exact-model router can register the service.
Other GET paths return HTTP 404. Every GET and POST response the handler
writes, including errors, carries `X-Clef-Identity-Sha256` (SHA-256 of the
identity as sorted-key, compact UTF-8 JSON), `X-Clef-Revision`,
`X-Clef-Checkpoint-Sha256` and `X-Clef-Tensor-Cache`. The startup readiness
line includes the same `identity_sha256`.

The claim has these limits:

- On a TT tensor cache hit the backbone runs on tensors converted by an earlier
  process. Their bytes are not verified, and `tt_tensor_cache` is then
  `preexisting-unattested`. Only a start with an empty revision cache converts
  the served tensors from the verified files.
- A checkpoint file changed after the startup check is not detected.
- Compiled TT-Metal kernels in `TT_METAL_CACHE` are not covered.
- The server process reports its own identity. This is not remote attestation
  of the host.

`clef-identity-check.py` exercises the manifest, checkpoint verification,
identity and HTTP surface on loopback with a fake engine and temporary files,
and checks the installed manifest against the pinned head and release code. It
is device-free: none of this has run on a physical P150 service.

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
lazy-weight pattern. At that point the Flash single-device loader was unchanged.
No kernels were prewarmed or KV caches allocated before readiness.

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
**first HTTP request**, not during startup. Those residual language-layer
loader costs motivated the derived-weight cache-hit change below; no
request-side KV or kernel work was moved ahead of readiness. See the
[bounded startup receipt](evidence/clef-startup-cache-hit-2026-10-03.json)
for package identities, individual paired timings, response checks, profiler
scope, preserved log paths, and resource limits.

### Derived-weight cache hits

For an existing dtype/layout- and revision-qualified TT tensor cache,
`pkgs/qwen36/cache-hit-fused-projections.patch` now loads fused GDN and
attention QKV tensors before constructing the full CPU fused weights.
`cache-hit-output-weight.patch` skips the full LM-head transpose in both
variants, and `cache-hit-single-projections.patch` skips the cached Flash
MLP, GDN, and attention projection transposes. Missing or unreadable caches
still take the original checkpoint-backed creation/repair path. The requested
mesh, tensor names and types, and device execution are unchanged.

In another serial, **one-process-per-arm** physical comparison of the prior
shard-cache baseline against this full change, process launch to `stage:ready`
was **74.039 → 35.943 seconds** for Clef 27B P150x2 (51% lower), and
**69.102 → 36.960 seconds** for Clef-Flash 9B P150 (47% lower). Each
baseline/candidate pair shared its revision-qualified TT tensor cache and
used the same private `/datapool` kernel-cache location. All **13 parsed HTTP
response objects per process** matched exactly between packages, including
probabilities and usage: short and 1,637-token long text, real PNG image,
four-frame video, changed/restored state, and joint noul/score requests.
The wrapper emitted readiness before any of these requests; their first-use
kernel compilations remained after readiness. No first-request or warmed
request latency improvement is claimed from these single samples.

This is a warm TT-tensor-cache startup result, **not** a full-model cold
rebuild, statistical startup distribution, or simulated GDN numerical gate.
A separate small TTNN 32×32 cache smoke covered creation, a matching hit,
and unreadable-flatbuffer repair; the corresponding host TTNN setup did
initialize and close the UMD driver, but it did not explicitly open a mesh.
The pinned TTNN 0.77 `as_tensor` cache-hit implementation checks file
presence/readability, not the source shape claimed by the older 0.74
serialization report. See the
[bounded derived-weight receipt](evidence/clef-derived-cache-hit-2026-10-03.json)
for exact packages, request timings, response examples, retained local logs,
and limits.

### Reference vision checkpoint initialization

A physical profile of the current cache-hit package measured **54.664 seconds**
from process launch to readiness on Clef 27B P150x2. Within that instrumented
startup, `from_pretrained` occupied 23.614 seconds, the separate CPU reference
visual load 4.465 seconds, and the subsequent TT vision initialization 6.932
seconds. These phases have different nesting and the profile is not directly
comparable to the preceding unprofiled 35.943-second sample. The language
backbone remains the larger startup phase.

The pinned checkpoints already contain all 333 reference-visual parameters in
BF16. The reference visual now constructs on the Torch `meta` device and
assigns those existing checkpoint tensors instead of initializing an FP32
vision tower, casting it to BF16, then copying the same weights into it. Its
nonpersistent rotary-frequency buffer is initialized and cast as before.
Host comparisons on **both** checkpoints found all 333 parameters and the
rotary buffer exactly equal to the previous loader, with no meta tensors left.
Direct host visual-loading times were approximately **4.001 → 0.034 seconds**
(27B) and **3.972 → 0.042 seconds** (Flash). TT vision/device construction,
checkpoint identities, GDN arithmetic and request-side work are unchanged.

With the same revision-qualified TT tensor caches and a private `/datapool`
`TT_METAL_CACHE`, serial physical wrapper startup pairs measured
**65.880 → 39.461 seconds** for 27B and **41.694 → 35.400 seconds** for
Flash. Reverse-order, startup-only pairs measured candidate/baseline
**26.981/28.914 seconds** for 27B and **65.719/34.216 seconds** for Flash.
The large first 27B difference cannot be assigned to this roughly
four-second CPU change, and Flash's reverse-order sample does **not**
demonstrate a consistent process-level speedup. Its logged model-config and
language-layer phases slowed before visual loading. These four small pairs
show substantial host/startup variability, not a latency distribution.

Both packages returned HTTP 200 with **13 exactly equal parsed response
objects per variant**: 132-token short, 1,637-token cross-chunk text, real
32×32 red PNG, four-frame video, changed/restored state, and joint noul/score,
including probabilities and usage. First/restored states also matched within
each process. First-use kernels compiled after readiness; later arms shared
the same binary-cache directory, so neither first-request nor warmed HTTP
acceleration is claimed. All meshes closed. The
[bounded visual-initialization receipt](evidence/clef-visual-meta-startup-2026-10-03.json)
records the packages, startup orders, individual samples, profile, retained
local logs and limits.
