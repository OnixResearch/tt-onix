# tenstorrent.nix

A standalone Nix flake for using Tenstorrent hardware with Nix/NixOS.
The `OnixResearch/tenstorrent.nix` `main` branch is the sole repository
authority; no external upstream remote is configured.

**Publication pending:** this repository is not yet cleared for a public
release. Tracked physical evidence and Git history include machine-specific
metadata; copyright, third-party patches, and binary fixture redistribution
also require owner review. See the [publication review](docs/publication-review.md)
before making any branch, tag, source archive, or build artifact public.

## License

Repository-authored code, documentation and Nix recipes without a separate
license notice are offered under [GNU LGPL version 2.1 only](LICENSE)
(`LGPL-2.1-only`).
Existing Apache-2.0-marked TT-Metal/Qwen patch portions and the
`pkgs/ttwkv7/architecture-check-descriptors.h` fixture retain their
[Apache-2.0 terms](LICENSES/Apache-2.0.txt). Fetched upstream packages,
dependencies, model artifacts, and saved binary fixtures retain their own
terms; this repository license does not grant redistribution rights to them.
See the [publication review](docs/publication-review.md) before distributing
the current tree or its Git history.

## Overlay

```nix
nixpkgs.overlays = [ tenstorrent.overlays.default ];
```

Provides top-level `tt-umd`, `tt-logger`, `tt-metal`, `tt-smi`, `tt-burnin`,
`tt-topology`, `tt-system-tools`, `luwen`, `qwen38`, and `nanbeige43`. It extends every Python
package set with `pyluwen`, `tt-flash`, and `tt-tools-common`. It also extends
kernel package sets with `tt-kmd`.

## NixOS module

```nix
{
  imports = [ tenstorrent.nixosModules.default ];
  hardware.tenstorrent.enable = true;
}
```

Matches the upstream `hardware.tenstorrent` module, replaces it when both are
present, and applies the overlay automatically.

## Fast local iteration

Enter the default development shell for Rust, Nickel, ShellCheck, TT-SMI, and
the reviewed training-platform tools:

```sh
nix develop
# Equivalent named shell:
nix develop .#training-platform
```

The shell clears ambient Python and TT-Metal runtime overrides. It does not
install or load a kernel driver. A NixOS activation remains necessary for
TT-KMD changes.

Package-only changes do not need a NixOS activation. Run the exact package from
the current checkout:

```sh
nix run .#ttwkv7 -- shape-test
nix run .#ttwkv7-constant-probe -- self-test
nix run .#ttwkv7-data-movement -- self-test
nix run .#training-platform-core -- self-test
```

Each command rebuilds only the affected derivations. Nix reuses the unchanged
host binaries, kernels, and dependencies from the store.

See [Model target consumer identities](docs/model-target-consumer-identities.md)
for the canonical profile edit, regeneration, and freshness workflow.

See [Qwen3.6-A3B candidate target](docs/qwen3-6-a3b-candidate.md) for the
metadata-only profile, bounded simulator verdict, and explicit non-claims.

See [Qwen3.8-27B adoption](docs/qwen38-27b-adoption.md) for the historically
validated P150x2 output and its guarded NixOS example. B=8 batch admission was
re-measured on the fused GDN step route and admitted on 2026-10-21 (per-user
retention 0.926, per-user TTFT ratio 4.52); B=16 is admitted under the same
per-user TTFT statistic (8.60 against 9.0, with the serialized-prefill group
maximum recorded as a non-gating diagnostic), and continuous-slot, prefix-cache, MTP,
and batched production serving still require their own physical
qualification.

The B=8 v3 interleaved-protocol receipt is not carried by any servable
package. It names execution identity `l4arf461…`, which no build in this tree
reproduces, while the served package exports the generic B=8 profile as
`…-b8-v2`. The v3 admission must therefore be re-earned on a buildable
identity before any package may carry it; the predeclared procedure is
`.cairn/changes/fix-qwen38-receipt-bootstrap/evidence/re-earn-procedure-2026-10-22.md`.
Keeping mutable receipt bindings out of the execution identity is what makes
that promotion satisfiable (`.cairn/changes/fix-qwen38-receipt-bootstrap/`).

Use `nix build .#qwen38` for the Qwen3.8-27B package. Its commands use the
`qwen38` prefix, including `qwen38-check` and `qwen38-p150x2-serve`.
The `qwen36` package and command names remain compatibility aliases.
The implementation directory `pkgs/qwen36`, upstream Python imports, cache
paths, and historical receipt schemas retain their existing names.
See the [staged decode comparison](docs/qwen38-staged-decode-comparison.md)
for the latest bounded timing result and its non-claims.
See [the fused GDN decode step](docs/qwen38-fused-gdn-decode-step.md) for the
default decode route, its measured 1.30x single-stream gain, and how to opt out.
See [Qwen3.8 structured reads](docs/qwen38-structured-reads.md) for the
`POST /v1/systemone` decision endpoint ported from vLLM PR 57250, its batched
read traces, and measured accuracy and latency.

See [Cloudflare Clef on Tenstorrent](docs/clef-tenstorrent.md) for the
separate pinned Clef (27B, P150x2) and Clef-Flash (9B, P150) joint-head
`/v1/systemone` services and their bounded physical smoke.

Use `nix build .#nanbeige43` for Nanbeige4.3-33B (revision
`c1093c501cc6460427c0ac634db7223d0450c0ee`, a plain `LlamaForCausalLM`) on the
P150x2 `1x2` mesh. It runs the runtime-matched TT-Metal 0.77 `tt_transformers`
generic path with TP2, the all-BFP8 mix by default (BFP8 attention, KV and MLP;
`--preset accuracy` selects BF16 attention and KV with a BFP8 MLP), and
`PAD_MLP_CORES=16`. `nanbeige43-check` runs the device-free
checks and admits a checkpoint by SHA-256. `nanbeige43-reference` writes the
CPU reference. `nanbeige43-reduce-checkpoint` derives reduced-layer checkpoints.
`nanbeige43-p150x2 compare|generate|bench` is the production command,
fail-closed on PCIe devices 0 and 1. `generate` accepts the same `--batch` sizes
as `bench` (P38): every row decodes together through the admitted path, all rows
produce identical ids for identical prompts, and generation stops when every row
reaches a stop token. Its decode rate is timed from the second step, so the
one-time decode-trace capture is excluded; batch 4 measured 14.47 tokens/s per
user (57.89 aggregate) against 12.67 at batch 1. The full curve is batch 1
12.67 per user, batch 4 14.47/57.89 aggregate, batch 16 14.48/231.66, batch 32
14.29/457.13 (P39), with ids byte-identical to batch 1 at every size.
`generate --expect-output-from <receipt>` (P40) makes that id equality a gate
inside the command instead of an ad-hoc comparison: it fails closed with an exit
code when the ids diverge. `generate --generate-runs N` (P77) measures the completion
more than once and records per-run rates plus their median, because a single generate
run is not reproducible: two consecutive identical batch-32 runs differed by 18.5 %
(P76), while three batch-1 runs differed by 0.8 %.
`bench --cache-probe {prefill,decode}` reads layer 0's paged K and V back after
prefill, or after one decode step, and records per row the page-table block, the
slot and the first two values at that slot for every kv head of both tensors. It is
the instrument the batch-8 investigation used (P42-P52): V is correct on every row
while K is stale for rows 5-7. It reads state back, writes nothing, runs after the
measured runs and their gates are recorded, and is absent unless requested.
The batch-8 defect, its elimination record and the priced fix path are documented in
`docs/nanbeige-batch8.md`; batch 8 is refused, and a device-free check enforces that.

On 2026-10-25 the full model passed the
teacher-forced gate on physical devices 0 and 1: top-1 92.2 %, top-5 99.0 %
over 512 positions, prefill PCC >= 0.9969. Decode capture and replay is the
default; `--no-trace` selects eager decode. Greedy sampling argmaxes the device
logits and reads back one index per row (`--no-device-argmax` keeps the host
path, which reads the gathered vocabulary replica); `compare` keeps the
full-logit path, and its accuracy metrics run on a single torch thread (P25),
which cut the batch-32 gate from 1363 s to 310 s with identical top-1/top-5 and
all 512 predicted tokens, at PCC deltas of at most 1.6e-14.

Pre-P8 shipped host-wall smoke numbers, superseded by the device-argmax block
below: batch-1 median 9.97 tokens/s (min 9.36; O1
9.69) with 128/128 ids equal to O1 and a median first token of 0.21 s; batch 4
10.16 tokens/s per user with O1-identical rows; batch 16 160.81 tokens/s
aggregate at 10.05 per user; batch 32 313.55 tokens/s aggregate at 9.80 per
user with 32/32 rows identical and ids equal to batch 16. Batch 32 is the
aggregate ceiling: batch 64 is rejected upstream before load
(`supported_batches = {1, 2, 4, 8, 16, 32}`). The host-path change did not
establish a batch-1 wall-clock gain: 9.966 vs O1's 9.694 is +2.8 % with a
9.36-10.86 spread, within run-to-run noise, and the earlier 11.178 was a single
measurement on the now-GC'd package. It is retained because it does strictly
less host work per step (one gathered replica, active-row/vocabulary slicing
before FP32 conversion, no dummy log-probs or cat) with byte-identical tokens,
not because it made decode measurably faster.

Device argmax is the shipped decode default (P8, 2026-10-27). An in-session
A/B at batch 32 measured 81.8 ms per step against 112.6-131.4 ms for the host
argmax over the gathered vocabulary, with both paths reproducing the admitted
batch-32 ids exactly and every row identical. The promoted package
`/nix/store/5j7n7bd9k870x3npwdjpdbhwdsr9dcf5-nanbeige43-0.1.0` then measured
386.18 tokens/s aggregate at batch 32 (samples 388.31/384.45/388.17/386.18/
384.40, per user 12.07) with ids exactly equal to the admitted batch-32
receipt, against 313.55 aggregate and 9.80 per user before the change on the
same protocol; batch 1 moved from 9.97 to 12.39 tokens/s per user with 128/128
ids equal to O1. The P6 split that motivated it: of a 108.51 ms step, 82.47 ms
was the traced device execution, 17.32 ms was the host argmax, and 6.5 ms plus
1.3 ms were enqueue and `to_torch`; the host-side FP32 copy inside the argmax
turned out to be worth only 1 ms (P7, reverted). The teacher-forced and prefill
gates are inherited from the untouched `compare` path. These remain host-wall
smoke measurements, not serving benchmarks.

Prefill is eager. Traced prefill was implemented and tested at the P300 [128,
1024] buckets: it reproduced the accepted numerical gate exactly (top-1
0.921875, top-5 0.990234375, corpus prefill PCC 0.99703, fixed-prompt PCC >=
0.99692) and the O1 ids, and 4096 stayed eager, but it lost at both traced
buckets (775.2 vs 795 tokens/s at 128; 2401.5 vs 2426 at 1024), so it was
reverted and is not the default. Prefill at these sizes is dominated by weight
streaming, so removing host dispatch saves little and replay adds its own
input-copy overhead. The 4096 path remains eager and unchanged at 2520 tok/s
(baseline 2517); the 0.1669 s batch-1 first token measured under the rejected
package is not a shipped claim.

Batch 8 is admitted by upstream `supported_batches` but corrupts rows 5-7
deterministically on this path; it is a supported-size correctness defect, not
an unsupported-size refusal, and stays refused.

That investigation is complete and recorded in
[the batch-8 record](docs/nanbeige-batch8.md). The cause is localised to the
qk-fused decode path, which places K alone on core row 1 at batch 8 while Q and V
share row 0, and that placement is load-bearing for the bounding-box
runtime-argument scheme. Forcing the non-fused path makes batch 8 correct (rows
identical, ids equal to the admitted stream) at 101.77 tokens/s aggregate, against
113.8 for two batch-4 streams, so the fix has no throughput case and is recorded
rather than applied. No batch-8 throughput or serving claim is made.

Batched prefill for `batch > 1` is available as an explicit opt-in
(`--batched-prefill`) and is not the default. Measured on 2026-10-26 against
the admitted batch-32 receipt, it prefills all 32 rows in one pass at 1427
tok/s and cuts first-token latency from 5.141 s to 2.745 s (1.87x against the
admitted figure, 2.13x against a same-session sequential control at 5.839 s).
It is admitted as a second configuration with its own receipts (P79), not as the
default, because its greedy ids differ from the sequential path's at token 2 - the
batch-variant prefill reduction order upstream cites for tt-metal#47238, not a runner
row-handling bug. On the shipped package at batch 32 with `--page-blocks 1024` it
measures TTFT median **3.098 s**, decode 439.34 against 440.99 aggregate (unchanged
within noise), all runs and rows identical,
and accuracy within 1e-4 on prefill PCC with teacher-forced top-1 0.904297 against
0.916016 (-1.17 points) and top-5 0.994141 against 0.992188. The pool requirement is
`--page-blocks` at or below **4096**, which is the same per-row context the sequential
default has at this batch size: 2048 measures 2.99 s and 4096 measures **2.66 s** (P85),
while 8192 dies with a 469 MB DRAM allocation failure. Accuracy at 4096 is identical to the 1024 screen (top-1
0.904297, top-5 0.994141), the pool changes latency and context but not quality, and
`generate` passes all its gates there at 3.63 s to first token with identical ids across runs
(P86). So at batch 32, `--batched-prefill --page-blocks 4096` is the recommended
configuration. The mode is now compared **paired**: `--prefill-modes both`
alternates the two paths run by run inside one session (P83), which removes the
session-to-session variation that made earlier ratios unreliable. At batch 32 with
`--page-blocks 1024` and ten alternating runs, batched prefill measures **4.35 s** against
**15.27 s** sequential in one session and **2.90 s** against **8.13 s** in a repeat - a
paired ratio of **2.81-3.51x**, i.e. about 3x - with ids identical within each mode and, at
`--page-blocks 4096`, no context cost. Earlier
unpaired figures (1.61x, 4.25x) are superseded, and the absolute medians are
session-dependent, so the ratio is quoted as a range. Sequential TTFT at this batch size also
carries a heavy tail that has reached 52 s (P82), and pool size is a weak lever (1.17x at
1024 against 8192) rather than the 2.6x once reported. The `accuracy` preset, the `batch > 1`
default prefill path, and every admitted number are unchanged.

A replacement profiler run named all 5,100 traced decode rows: matmul
operations held 90.6 % of the 76.65 ms per-replay-device kernel sum, with
collectives 4.3 % and SDPA 0.7 %. The matmul bucket's 290 GB/s is close to what the
machine sustains at this model's own per-matmul weight sizes: a decode-shaped
matmul reads at 271 GB/s with a 42 MB weight and 264 GB/s with 84 MB, rising to
348-367 GB/s only for per-op weights of 250 MB to 1 GB, which this model does not
have except for the once-per-step LM head (P63). Larger-N probes (P62) reported
357-392 GB/s, but those ops were 6-24 times larger than any in this model, so that
headroom reading was a scale artifact and is withdrawn. No material streaming
headroom exists at these op sizes: per-op timing at the model's 42 MB is not stable to
better than about ±20 % (154-1537 µs against a 183 µs median), so per-op probes cannot
resolve a difference between configurations, while the profiled 290 GB/s bucket over
5,100 ops sits at or above the 230-270 GB/s band those probes show. No supported
configuration change is demonstrated to close the gap to the 512 GB/s pin rate, and the
bucket-level figure is the reliable one. The matmul bucket still mixes attention, MLP,
and LM-head work, so no full four-way device attribution is claimed. The
supported prefetcher is closed by L1 arithmetic (minimum ring 115 unpadded or
118 padded, against reachable rings up to 80), so no supported
precision-preserving decode knob remains; a prefetcher entry or hand-built
DRAM-sharded program config would need its own declared experiment. All-BFP8
passed the TP2 accuracy screen, but its O5 bench median was 5.71 tokens/s
(required > 9.69), so it was not promoted. That run is not evidence that lower
precision costs speed: `accuracy` already runs a BFP8 MLP, so the mix changes
only WQKV/WO and KV and cuts streamed bytes by 14.2 %; it also drops attention
from HIFI4 to HIFI2, and the same run's decode samples spanned
8.456/5.715/4.985/4.279/7.682 tokens/s while its prefill was above accuracy at
1024/4096 and below at 128. `accuracy` remained the default until P9 below.

The all-BFP8 mix is now the default (P9, 2026-10-27). O5 had declared it
promotable once its speed gates passed and its accuracy held; P8's device argmax
made the speed measurable. On the promoted package
`/nix/store/15msnna7291jxyay9iznl94wfajdqb9j-nanbeige43-0.1.0` it measures
13.1-14.3 tokens/s per user at batch 1 (13.14-14.32 across two sessions of the
same code path), with a 0.153-0.361 s first token and 537-890 / 2545-2824
tok/s prefill at 128/4096, and 430.5-443.9 tokens/s aggregate at batch 32
(13.45-13.87 per user), against 12.39 and 386.18 for the previous default on
the same protocol. The full admitted sweep on the promoted package is batch 1
13.14-14.28, batch 2 28.47 aggregate at 14.24 per user (first token 0.341 s), batch
4 56.91 aggregate at 14.23 per user, batch 16 225.93 at 14.12, and batch 32
443.90 at 13.87 tokens/s, every size with rows identical across runs and its
receipt gates passed. An earlier rebind on a previous package measured 445.11 at
13.91 per user; the currently shipped package was rebound across all five sizes
(P57/P59) at 13.98 (b1), 27.27 (b2), 57.25 (b4), 227.03 (b16) and 440.99 (b32)
aggregate, 13.78 per user at batch 32, every run with rows identical, outputs
identical across runs and ids exactly equal to its admitted receipt with no first
divergence. Those single-run rates bracket the admitted figures on both sides,
which is the shape of run-to-run noise rather than a regression. The pre-P8
figures were 9.97, 37.23/10.16, 160.81/10.05 and 313.55/9.80. The recorded cost is accuracy:
teacher-forced top-1 0.921875 -> 0.908203 (-1.37 points) and top-5 0.990234 ->
0.994141, corpus prefill PCC 0.996854, fixed prompts >= 0.9974, all inside the
retained gates. The default's accuracy is also bound at batch 32 (P26): top-1
0.916016, top-5 0.992188, corpus PCC 0.996854, fixed prompts
0.998948/0.998381/0.997452, all 32 rows identical, so the mix carries a measured
batch claim rather than an accuracy-only one. Both accuracy screens were re-run on
the currently shipped package (P60) and reproduce bit-identically - batch 1 top-1
0.908203/top-5 0.994141, batch 32 0.916016/0.992188, and the same four prefill
PCCs - so the shipped identity is numerically identical to the recorded one for
its decode tokens and its accuracy alike. Multi-row first-token latency under the mix is prefill-bound and scales with rows,
pooled over all admitted bench receipts: batch 1 median 0.169 s (n=46), batch 2 0.346
(n=6), batch 4 0.650 (n=21), batch 16 2.515 (n=16) and batch 32 **5.00 s** (n=46, p90
7.81, range 4.43-16.10). At batch 32 that is 269 tokens/s of sequential per-row prefill,
matching the P5 control. The earlier "bimodal" description was a tail rather than a second
mode: 5 of 46 batch-32 samples (10.9 %) exceed 1.5x the median at 7.8-8.4 s, plus one
16.1 s sample from a short-warmup protocol whose cause is not established (P78). `--preset
accuracy` keeps the previous quality for anyone who needs it. These remain
host-wall smoke measurements, not serving benchmarks.

The upstream `performance` preset and the tested single-card BFP4 mixes failed
accuracy retention. No tested mix both passed accuracy and fit on one card; no
independent two-instance DP run or DP throughput is claimed, and unexercised
DP wrappers are not shipped. Fixed-prompt comparisons retain all batch rows and
gate on worst-row prefill PCC; historical O3 fixed-prompt receipts covered row
0, while corpus gates covered every row. These are host-wall smoke
measurements, not a serving benchmark, and there is no serving endpoint. See
`.cairn/changes/bring-up-nanbeige43-33b-p150x2/` for design, receipts, and
non-claims.

To benchmark OMP `task` subagents on the running P150x2 Qwen3.8-27B service,
run `python3 scripts/qwen38-subagent-benchmark.py --timeout-seconds 600 --output /tmp/qwen38-subagents.json`.
The bounded three-worker workload compares three sequential task calls with
one concurrent task batch. Configure OMP's `mesh/Qwen3.8-27B` provider at
`http://127.0.0.1:9337/v1` first. The runner checks parent and child model
identities, probes the gateway's physical PCIe IDs, and saves the OMP event
streams next to the JSON receipt. The probe does not independently attest each
child; paired wall times on a shared host are not throughput or model-quality
admission evidence.

One paired run on 2026-09-29 took 198.07 s for the serial root and 110.39 s
for the parallel root (1.79× shorter wall time). All six child reviews returned
the requested findings and fixes. Child durations summed to 71.84 s serial
versus 122.48 s parallel; this is not a per-worker inference-speed gain. The
serial and parallel parent prompts differ, and one shared-host pair does not
establish a stable throughput improvement.

See [Qwen trace-reuse optimization](docs/qwen-0.77-trace-optimization.md) for
an opt-in B=4 route with measured request-time savings and unchanged service defaults.
The later [binding-identity optimization](docs/qwen-binding-identity-optimization.md)
reduces accessor overhead without skipping binding checks.
The [readback evaluation](docs/qwen-readback-evaluation.md) retains a rejected
candidate and the bounded B=1 component measurements.
The [staged GDN evaluation](docs/qwen-gdn-staged-evaluation.md) records the
default-off `norm-state` candidate, its B=1 sequence check, and remaining limits.

See [Qwen3.8 optimization research](docs/research/qwen38-p150x2-2026-09-04.md)
for the source review, historical evidence limits, and the original nine Cairn packages.
The research does not claim a new hardware result or deployment.

See [the OrcaRouter checkpoint-switch plan](docs/research/qwen38-orcarouter-switch-2026-09-04.md)
for the proposed native-weight route to Qwen3.8-27B-Uncensored on P150x2.
The linked GGUF release is not a direct input for the current TT-NN service.

Build a stable result link when you need to inspect or run several commands:

```sh
nix build .#ttwkv7 -o result-ttwkv7
./result-ttwkv7/bin/wkv7 shape-test
./result-ttwkv7/bin/wkv7-constant-probe self-test
```

Run the focused device-free checks before the final host integration check:

```sh
nix build .#checks.x86_64-linux.development-interface
nix build .#checks.x86_64-linux.ttwkv7-architectures
```

A NixOS activation is still necessary for kernel modules, udev rules, system
environment changes, and service changes. Use the full host closure as the
final integration gate, not as the package inner loop.

## Repository quality

Run the complete device-free quality gate from the repository root:

```sh
nix run .#quality
```

The command uses pinned Rust, shell, Python, Nix, Nickel, and Cairn tools.
It does not acquire a Tenstorrent device. See
[`docs/repository-quality.md`](docs/repository-quality.md) for the parser and artifact workflows.

The launcher bootstraps `.#nix-patched-build-remote` locally, then passes its
matched Nix 2.36.0 `__build-remote` hook to the normal evaluation and aggregate
build. Only the bootstrap disables builders; the gate retains configured remote
builders and plugin files. The minimal repair hashes an overlong slot-lock key
after `ENAMETOOLONG`, without changing transport credentials, short-URI lock names
or the existing upload-lock protocol. The [remote proof](pkgs/nix-patched-build-remote/evidence/2026-10-23/remote-proof.json)
records an actual forced gRPC build, not a local fallback or connectivity probe.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the contribution and evidence policy.

## Onix packages

The flake publishes host-matched `ttsim` runtime bundles on `x86_64-linux` and `aarch64-linux`.
See [`docs/ttsim-runtime.md`](docs/ttsim-runtime.md) for wrappers, the bounded smoke, and simulator non-claims.

### Isolated developer evaluation tools

`tt-lang-sim` **1.1.6** is the upstream pure-Python TT-Lang simulator, not
`ttsim`, a TT-Metal device runtime, or production `ttnn`. Its explicit Python
closure clears ambient `PYTHONPATH`/`PYTHONHOME` and user-site imports. Only the
simulator and statistics commands are exposed; runtime SFPI/toolchain setup or
downloads are not part of this package.

```sh
nix build .#tt-lang-sim .#tenstorrent-official-skills .#tenstorrent-eval-tools
nix run .#tt-lang-sim -- /absolute/path/kernel.py --grid 1,1 --num-devices 1 --trace /tmp/kernel.jsonl
nix run .#tt-lang-sim-stats -- /tmp/kernel.jsonl
nix run .#tenstorrent-eval -- --help
```

The separate `tenstorrent-eval-tools` package/app is exposed on
`x86_64-linux`; its explicit Python closure includes offline scoring
dependencies. The simulator and official reference package do not depend on
the production model package.
It provides `traffic` and `retention freeze/score/compare/collect`; see
[the evaluation tool guide](docs/tenstorrent-eval-tools.md) for modeled
tile/exponent/L1 accounting, frozen tokenizer identities, full-vocabulary
teacher-forced scoring and provenance-matched comparisons. Offline scoring is
not full-model evidence. The separate collector requires the hardware owner's
package-matched runtime, checkpoint, mesh and device isolation; the ordinary
installed offline wrapper does not load production TT-NN.

The installed simulator passed an actual one-tile add and matmul with exact
float32 outputs, and the upstream statistics command reported four DRAM tile
reads and two writes. These are **modeled** tensor transfers, not measured
hardware traffic, timing, placement, or device numerical fidelity. Default
reduced-precision simulation promotes arithmetic to float32; use upstream
`--no-float32-promotion` when investigating native dtype behavior, and still
require silicon qualification. The bounded receipt is in
[the integration Cairn change](.cairn/changes/isolated-tt-lang-sim-official-skills/).

`tenstorrent-official-skills` pins the official
[`tenstorrent/skills`](https://github.com/tenstorrent/skills/tree/b6752095f329a04bf6573fb4022b13bc45799167)
revision **b6752095f329a04bf6573fb4022b13bc45799167**. It supplies read-only
review-core, precision, trace, L1 and TTNN kernel domains, complete local
reference directories, and upstream license/NOTICE files. The CLI automatically
loads **review-core first**, then **at most two distinct domain skills** per
invocation (repeated domain names are deduplicated):

```sh
nix run .#tenstorrent-official-skills -- list
nix run .#tenstorrent-official-skills -- resolve review-core precision
nix run .#tenstorrent-official-skills -- read review-core kernel
nix run .#tenstorrent-official-skills -- read l1 --file references/data-movement-tiers.md
```

This reference reader does not install a marketplace/plugin, execute bundled
scripts or hooks, enable telemetry, mutate the home directory, or refresh the
older `.pi/skills` pins. Reading a reference is not authorization to execute
its suggested commands. These references are not package-matched TT-Metal API
authority.

New kernel candidates remain opt-in and default-off:
`TT_DECODE_MATVEC=1` selects the native decode strip matvec, and
`TT_WKV7_STATE_VIEW=1` selects direct eligible recurrent-state views through
the existing native model gates. The Qwen package supplies
`qwen38-rounded-register-swiglu-check --semantic` (device-free) and
`--device --batch 1 --device-id 0 --iterations 3` (physical hardware required;
`--batch 8` is also supported). `QWEN38_ROUNDED_REGISTER_SWIGLU=1` selects the
decode-only candidate in the real MLP path; it does not admit fused gate/up
projection or change service defaults. For example, with exclusive device
ownership and the existing local checkpoint prerequisites:

```sh
TT_DECODE_MATVEC=1 TT_WKV7_STATE_VIEW=1 nix run .#metalium-rs-rwkv7-model -- --fast
nix build .#qwen38 -o result-qwen38
./result-qwen38/bin/qwen38-rounded-register-swiglu-check --semantic
./result-qwen38/bin/qwen38-rounded-register-swiglu-check --device --batch 1 --device-id 0 --iterations 3
```

The 2026-10-23 physical card-0 native gate passed all four kernel tests,
all-12-layer checkpoint/oracle comparisons with neither/either/both flags, and
three-token continuation with both flags. The strip-matvec component medians
were near-flat (some slightly slower), so it is **not promoted for performance**.
The eligible direct state view removed the nine compact-copy transfer packets
in its bounded consumer experiment (817→807 μs), not a model-throughput claim.
See [the native receipts](.cairn/changes/native-decode-strip-matvec-view-consumers/evidence/decode-matvec.md).
The normal native package check retains the full debug-assertion workspace tier
and optimizes its host simulator plumbing and independent numerical oracles with
`CARGO_PROFILE_TEST_OPT_LEVEL=3`. The original 1500-second timeout ended at the
Lanczos3 adjoint's dense Flex impulse oracle, not a reported assertion failure.
The source-matched optimized interpolation binary passed all **14 tests in
116.45 seconds**, including the original adjoint and negative controls; debug
assertions, integer overflow checks and profile-sensitive sample coverage remain
enabled. This scoped proof does not substitute for the full normal package gate.
See [the profile diagnosis](metalium-rs/docs/learnings/performance-log.md#native-package-check-profile-2026-10-23-ttsim).

The full normal package gate then **passed in 937.98 seconds** on the configured
gRPC builder using the matched repaired hook: **1,556 passed, 0 failed, 46
existing ignored, 0 filtered**, across 258 runtime binaries and 14 doctest
crates. All 14 interpolation tests passed there in 28.86 seconds. The original
1500-second bound and full workspace/debug-assertion coverage were unchanged;
the separate failed local four-job compilation attempt remains in the
[qualification record](.cairn/changes/isolated-tt-lang-sim-official-skills/evidence/installed-tooling-2026-10-23.json).

The rounded-register Qwen candidate passed exact B=1/B=8 physical numerical
parity after matching native BF16 pack rounding, but its installed bounded
component medians were about **2.3× slower at B=1 and 4.2× at B=8**, so it is
**not promoted**. The unchanged installed Qwen checker now also passes exact
BF16 comparison at B=1/B=8 on source-built TTsim 1.10.6 with the bounded
Blackhole dense-L1 zero-flag model extension. This is simulated single-chip
component acceptance, not full-model, multi-chip, physical or performance
admission. The separate GDN simulator limitation remains. The checker
also provides `--model-device --batch 8 --iterations 2` for bounded two-card
continuation, with explicit local model path and decode profile; see
[the Qwen candidate guide](docs/research/qwen38-rounded-register-swiglu.md)
for exact geometry, arithmetic and command prerequisites. These candidate checks
do not confer serving admission.

Fresh content-bound B=1 collection on physical devices 0 and 1 scored the
original six passages over all **248320 vocabulary columns**. Baseline and
decode-only treatment both had mean NLL **0.7451044766709086**, with all six
passage deltas exactly zero; the original **+0.010-nat diagnostic** passed.
The collector bound the actual weight index and all 66 referenced shard
contents, rather than treating a mutable model path or declared checkpoint
label as proof of weight identity. See [the retention receipt](pkgs/qwen36/evidence/2026-10-23/rounded-register-retention.json).
Separately, eight real prefills plus two recurrent continuation steps ran on
both cards; that is a B=8 route smoke, **not B=8 retention or throughput
acceptance**. No global quality floor, performance promotion, or serving
admission follows. Qualification left the Qwen service stopped.

Simulator arithmetic, CPU semantics and modeled byte
counts alone do not establish hardware speedups, full-model retention, or
production acceptance.

The [TT-Metal 0.77 candidate](docs/tt-metal-0.77-port.md) builds, but its GDN numerical gate remains blocked. Do not use this candidate as production acceptance evidence.

On `x86_64-linux`, the flake also publishes:

- `rwkv-lab` and `rwkv-layer-harness` for deterministic device-free RWKV-7 validation;
- `rwkv7-p150x2-runtime` and `rwkv7-p150x2-evidence` for replicated full-model two-card validation;
- `ttwkv7` plus bounded and persistent operator wrappers;
- retained partial and passing ttWKV7 evidence validators;
- `metalium-rs` plus its RWKV-7 WKV7, layer, model, tokenizer, generate, and prompt gates;
- `ttwkv7-owner-control` and `tt-vibethinker-bench`.

The focused checks include positive and negative fixtures and do not acquire a
Tenstorrent device. The `metalium-rs-rwkv7-tokenizer` gate is device-free and is
part of both `nix flake check` and `nix run .#quality`; the other native-stack
model gates require a card. The P150x2 runtime uses full-lockstep replication. It does not
claim a two-card speedup or fresh physical execution. Building `ttwkv7` requires
allowing its unfree package license. See
[`docs/migration-provenance.md`](docs/migration-provenance.md) for the exact source
boundary and non-claims.

## References

### Project and model references

- [onixcomputer/onix-core](https://github.com/onixcomputer/onix-core) — source repository for the migrated Onix package snapshot and its development history.
- [BlinkDL/RWKV-LM](https://github.com/BlinkDL/RWKV-LM) — upstream RWKV training and inference source used as an implementation reference.
- [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) — parent backend reference for RWKV-7, CUDA, Vulkan, and ggml operator work.
- [marty1885/llama.cpp](https://github.com/marty1885/llama.cpp) — package upstream for the Metalium backend and experimental RWKV trees.

### Qwen3.6-A3B model target references

- [Knight-Ops/tt-metal `qwen3_6_a3b-bringup`](https://github.com/Knight-Ops/tt-metal/tree/qwen3_6_a3b-bringup/models/demos/qwen3_6_a3b) — upstream tt-nn bring-up of Qwen3.6-35B-A3B for one Blackhole P150.
- [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) — candidate checkpoint and tokenizer source for the Qwen3.6-A3B model target.

### Qwen3.8-27B adoption reference

- [tenstorrent/tt-metal PR 53543](https://github.com/tenstorrent/tt-metal/pull/53543) — adopted Qwen3.8-27B-on-qwen36 change (closed-unmerged upstream). See [Qwen3.8-27B adoption](docs/qwen38-27b-adoption.md) and `pkgs/qwen36/authority.ncl` for the reviewed pin.
- [sjettTT/tt-qwen-3.8-flash-next](https://github.com/sjettTT/tt-qwen-3.8-flash-next/tree/main/models/demos/blackhole/qwen38_flash_next) — independent Blackhole MTP serving reference for the Qwen3.8 family (different 51.2B Flash-Next checkpoint). Its pinned numerics show width-k verify rows breaking near-ties differently from 1-row decode; the local screen records margins to classify any divergence.

### Tenstorrent platform references

- [tenstorrent/tenstorrent.github.io](https://github.com/tenstorrent/tenstorrent.github.io) — source for the public Tenstorrent documentation site.
- [tenstorrent/tt-metal](https://github.com/tenstorrent/tt-metal) — package-matched TT-Metalium, TT-NN, tests, and technical documentation source.
- [tenstorrent/tt-isa-documentation](https://github.com/tenstorrent/tt-isa-documentation) — instruction-set and hardware behavior reference.
- [tenstorrent/tt-low-level-documentation](https://github.com/tenstorrent/tt-low-level-documentation) — LLK, NoC, overlay, and kernel performance reference.
- [tenstorrent/sfpi](https://github.com/tenstorrent/sfpi) — packaged device compiler source and release metadata.

### Tenstorrent host and simulation references

- [tenstorrent/tt-umd](https://github.com/tenstorrent/tt-umd) — package-matched user-mode driver and host interface source.
- [tenstorrent/tt-kmd](https://github.com/tenstorrent/tt-kmd) — package-matched kernel driver and device-node source.
- [tenstorrent/luwen](https://github.com/tenstorrent/luwen) — package-matched chip discovery, diagnostics, and firmware communication source.
- [tenstorrent/ttsim](https://github.com/tenstorrent/ttsim) — device-free simulator source and ABI reference.

### Rust runtime references

- [brittonr/metalium-rs](https://github.com/brittonr/metalium-rs) — historical fork of Knight-Ops/metalium-rs: native Rust Blackhole firmware, `tt-kmd` host transport, Tensix kernels, and a Burn backend, independent of TT-Metalium. The `metalium-rs` package builds the in-tree `metalium-rs/` tree, which is the source authority; the fork remains the upstream and historical pin.

### Tenstorrent operations references

- [tenstorrent/tt-tools-common](https://github.com/tenstorrent/tt-tools-common) — shared behavior and schemas for Tenstorrent operations tools.
- [tenstorrent/tt-smi](https://github.com/tenstorrent/tt-smi) and [tenstorrent/tt-topology](https://github.com/tenstorrent/tt-topology) — diagnostics and topology sources.
- [tenstorrent/tt-flash](https://github.com/tenstorrent/tt-flash) — firmware update and compatibility-check source.
- [tenstorrent/tt-burnin](https://github.com/tenstorrent/tt-burnin) — hardware stress and validation source.
- [tenstorrent/tt-system-tools](https://github.com/tenstorrent/tt-system-tools) — incident collection and `tt-oops` source.

### RWKV ecosystem resources

- [Recently updated RWKV repositories](https://github.com/search?o=desc&q=rwkv&s=updated&type=Repositories) — GitHub discovery search for current RWKV projects.
- [RWKV Wiki](https://wiki.rwkv.com) — community documentation for RWKV architectures, models, terminology, and tools.
- [josStorer/RWKV-Runner](https://github.com/josStorer/RWKV-Runner) — desktop runner, model manager, and OpenAI-compatible RWKV service.
- [JL-er/RWKV-PEFT](https://github.com/JL-er/RWKV-PEFT) — parameter-efficient fine-tuning reference for RWKV models.
- [cgisky1980/ai00_rwkv_server](https://github.com/cgisky1980/ai00_rwkv_server) — Rust and Vulkan RWKV inference server with an OpenAI-compatible API.

### vLLM Tenstorrent plugin

- [tenstorrent/vllm-tt-plugin](https://github.com/tenstorrent/vllm-tt-plugin/tree/bef89e429e202caa38d3e3c8a24da4e8cc02405a) — standard vLLM platform plugin at `bef89e429e202caa38d3e3c8a24da4e8cc02405a`. This source targets vLLM 0.26.0 and replaces the deprecated Tenstorrent vLLM fork. See [the upgrade evaluation](docs/tenstorrent-upgrade-evaluation.md) for compatibility limits and local evidence.

### Qwen3.8 optimization research references

- [Adartras Qwen3.6-A3B bundle](https://huggingface.co/Adartras/qwen3.6-a3b-blackhole/tree/42a71032659a9d0e3b860b6104c59cfa6ca82149) — independent MoE reference for speculative sampling, recurrent-state commit, and paged-cache constraints. Its measurements are not local dense-model evidence.
- [TT-Metal 0.77 candidate](https://github.com/tenstorrent/tt-metal/tree/9f9cd4fd590f4b606bd0981a4fe0b6403eb38ec9) — immutable runtime candidate for fused GDN prefill. The local package pin remains unchanged.
- [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318) — algorithm reference for target-distribution preservation, not a local performance claim.

### OrcaRouter checkpoint references

- [Qwen3.8-27B-Uncensored-GGUF](https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored-GGUF/tree/a855f377abf5cbda99a278414466743f427e97c8) — requested model release. GGUF frontend support does not establish Tenstorrent placement.
- [Qwen3.8-27B-Uncensored BF16](https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored/tree/404ea47aaa5d8a8b00049c9e9750089aca011ab2) — recommended safetensors candidate for the existing TT-NN path. File access and local compatibility remain unverified.
