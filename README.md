# tenstorrent.nix

A standalone private Nix flake for using Tenstorrent hardware with Nix/NixOS.
The `OnixResearch/tenstorrent.nix` `main` branch is the sole repository
authority; no external upstream remote is configured.

## Overlay

```nix
nixpkgs.overlays = [ tenstorrent.overlays.default ];
```

Provides top-level `tt-umd`, `tt-logger`, `tt-metal`, `tt-smi`, `tt-burnin`,
`tt-topology`, `tt-system-tools`, `luwen`, and `qwen38`. It extends every Python
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

See [Qwen3.8-27B adoption](docs/qwen38-27b-adoption.md) for the validated
P150x2 path and serialized NixOS inference service.

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

## Onix packages

The flake publishes host-matched `ttsim` runtime bundles on `x86_64-linux` and `aarch64-linux`.
See [`docs/ttsim-runtime.md`](docs/ttsim-runtime.md) for wrappers, the bounded smoke, and simulator non-claims.
The [TT-Metal 0.77 candidate](docs/tt-metal-0.77-port.md) builds, but its GDN numerical gate remains blocked. Do not use this candidate as production acceptance evidence.

On `x86_64-linux`, the flake also publishes:

- `rwkv-lab` and `rwkv-layer-harness` for deterministic device-free RWKV-7 validation;
- `rwkv7-p150x2-runtime` and `rwkv7-p150x2-evidence` for replicated full-model two-card validation;
- `ttwkv7` plus bounded and persistent operator wrappers;
- retained partial and passing ttWKV7 evidence validators;
- `ttwkv7-owner-control` and `tt-vibethinker-bench`.

The focused checks include positive and negative fixtures and do not acquire a
Tenstorrent device. The P150x2 runtime uses full-lockstep replication. It does not
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

- [brittonr/metalium-rs](https://github.com/brittonr/metalium-rs) — fork of Knight-Ops/metalium-rs: native Rust Blackhole firmware, `tt-kmd` host transport, Tensix kernels, and a Burn backend, independent of TT-Metalium. The flake pins fork commit `d8c6e2fc4579cf29aa7525ddd086a90f81da2260` as the `metalium-rs` package.


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
