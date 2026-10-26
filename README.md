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
generic path with TP2, the upstream `accuracy` preset (BF16 attention and KV,
BFP8 MLP), and `PAD_MLP_CORES=16`. `nanbeige43-check` runs the device-free
checks and admits a checkpoint by SHA-256. `nanbeige43-reference` writes the
CPU reference. `nanbeige43-reduce-checkpoint` derives reduced-layer checkpoints.
`nanbeige43-p150x2 compare|generate|bench` is the production command,
fail-closed on PCIe devices 0 and 1. On 2026-10-25 the full model passed the
teacher-forced gate on physical devices 0 and 1: top-1 92.2 %, top-5 99.0 %
over 512 positions, prefill PCC ≥ 0.9969. Decode capture and replay is the
default; `--no-trace` selects eager decode. The host greedy path reads one
gathered replica and slices active rows and vocabulary before conversion;
`compare` keeps the full-logit path. On the delivered package the O0-protocol
traced batch-1 bench reached a median 9.97 tokens/s (min 9.36; O1 9.69) with
128/128 ids identical to O1 and a median first token of 0.21 s. The host-path change did not establish a batch-1
wall-clock gain: 9.966 vs O1's 9.694 is +2.8 % with a 9.36-10.86 spread,
within run-to-run noise, and the earlier 11.178 was a single measurement on
the now-GC'd package. It is retained because it does strictly less host work
per step (one gathered replica, active-row/vocabulary slicing before FP32
conversion, no dummy log-probs or cat) with byte-identical tokens, not because
it made decode measurably faster. Batch 32's 313.6 vs 160.8 aggregate is the
only established throughput improvement. `bench --batch 16`
reached 160.8 tokens/s aggregate at 10.05 tokens/s per user with every row
identical and ids equal to the admitted batch-16 receipt; it passed the
teacher-forced gate on every row and is not bit-identical to batch 1
(divergence at id 101). Batch 4 stayed O1-identical at 10.16 tokens/s per
user. `bench --batch 32` reached 313.6 tokens/s aggregate at 9.80 tokens/s
per user with 32/32 rows identical; its ids exactly match batch 16 and
retain the id-101 divergence from batch 1. `compare --batch 32` passed the
fallback gate on every row (worst top-1 0.9121, top-5 0.9961, prefill PCC
>= 0.99692). Batch 1 remains the default; batch 32 is accepted for aggregate
throughput. A replacement profiler run named all 5,100 traced decode rows: matmul
operations held 90.6 % of the 76.65 ms per-replay-device kernel sum, with
collectives 4.3 % and SDPA 0.7 %. The matmul bucket still mixes attention, MLP,
and LM-head work, so no full four-way device attribution is claimed. The supported
prefetcher is closed by L1 arithmetic (minimum ring 115 unpadded or 118 padded,
against reachable rings up to 80), so no supported precision-preserving decode
knob remains; a prefetcher entry or hand-built DRAM-sharded program config would
need its own declared experiment. Batch 8 corrupts rows 5–7 and
is refused. These batch claims apply to the
`accuracy` preset. The upstream `performance` preset and the tested
single-card BFP4 mixes failed accuracy retention. All-BFP8 passed the TP2
accuracy screen. Its O5 bench median was 5.71 tokens/s (required > 9.69), so it
was not promoted, but that run is not evidence that lower precision costs
speed: `accuracy` already runs a BFP8 MLP, so the mix changes only WQKV/WO and
KV and cuts streamed bytes by 14.2 %; it also drops attention from HIFI4 to
HIFI2, and the same run's decode samples spanned 8.456/5.715/4.985/4.279/7.682
tok/s while its prefill was above accuracy at 1024/4096 tokens
(2719/2753 vs 2548/2530) and below at 128 (726 vs 790). `accuracy` remains
default. No tested mix both passed accuracy and fit on
one card; no independent two-instance DP run or DP throughput is claimed,
and unexercised DP wrappers are not shipped. Fixed-prompt comparisons now
retain all batch rows and gate on worst-row prefill PCC; historical O3
fixed-prompt receipts covered row 0, while corpus gates covered every row.
These are host-wall smoke measurements, not a serving benchmark, and there
is no serving endpoint. See
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

See [CONTRIBUTING.md](CONTRIBUTING.md) for the contribution and evidence policy.

## Onix packages

The flake publishes host-matched `ttsim` runtime bundles on `x86_64-linux` and `aarch64-linux`.
See [`docs/ttsim-runtime.md`](docs/ttsim-runtime.md) for wrappers, the bounded smoke, and simulator non-claims.

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
