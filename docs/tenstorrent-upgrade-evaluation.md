# Tenstorrent upgrade evaluation

Review date: 2026-09-07 (local time).

## Result and scope

This change upgrades TTsim from 1.10.2 to 1.10.6. TT-Metal stays at 0.74.0 and TT-KMD stays at 2.10.0. No service, kernel module, or firmware changed on a production machine.

The evaluation covers three separate paths: simulator compatibility, driver build compatibility, and TT-Metal/model compatibility. Source review passes share one reviewer. They are not independent audits.

Acceptance requires package checks, a bounded simulated device open, firmware compilation, representative TT-NN operations, and a clean close. This smoke does not cover model weights, RWKV recurrence, Qwen MTP, or P150x2 execution. It does not authorize a hardware attempt.

## TTsim 1.10.6

The package retains its existing 13 host-specific assets. Nix downloaded each asset and computed its SHA-256 hash. Nix fetchers require these hashes. Runtime receipts retain BLAKE3 identities.

The baseline 1.10.2 smoke passed with TT-Metal 0.74.0. The candidate 1.10.6 smoke also passed with that runtime. Both opened one simulated Blackhole device, compiled firmware, ran `ttnn.ones` and `ttnn.add`, and closed the device.

The first candidate receipt exposed an existing defect: the validator always emitted simulator version 1.10.1. The library identity correctly identified 1.10.6, but the version field did not. This change aligns the validator with 1.10.6 and adds two checks:

- Package evaluation rejects a mismatch between the simulator pin and the validator declaration.
- The Nix smoke check compares the emitted version with the actual package version.

The validator rejects prior versions 1.9.2, 1.10.1, and 1.10.2. Malformed-receipt fixtures now name the current version, so each fixture still reaches its intended rejection.

The runtime checks retain negative cases for missing descriptors, wrong descriptors, missing ABI symbols, wrong host architecture, and unsupported targets. The multi-chip wrapper still rejects execution before device access.

### Upstream limits

The [1.10.6 README](https://github.com/tenstorrent/ttsim/blob/v1.10.6/README.md) still recommends slow dispatch and disables `SFPLOADMACRO`. Its multi-chip guidance now names `micah/native-sim`, not the older `mcraighead/mc-p300` branch. The packaged 0.74 runtime still lacks a reviewed Blackhole P300 descriptor.

The new `bh_x4` release assets remain outside the admitted asset set. No AArch64 runtime execution or multi-chip simulation formed part of this evaluation.

## TT-KMD 2.11.0

The [release](https://github.com/tenstorrent/tt-kmd/releases/tag/ttkmd-2.11.0) fixes TLB ownership, TLB locking, and device-removal races. Source inspection confirms the separate `tlb_mutex`, the new SMC ioctl, and the IOMMU requirement for firmware logs.

The isolated build candidate uses this source identity:

```text
repository: tenstorrent/tt-kmd
tag: ttkmd-2.11.0
unpacked SHA-256: sha256-Wjxz8kfRNeq0CpGn0M4ro6XnuyyUyYFV1Y6sOGHzkA0=
```

The candidate built successfully against Linux 6.18.34 from this repository's pinned inputs. Its output is `/nix/store/i08xr4i9riy1d8qcm1hv6z27a6abvgch-tt-kmd-2.11.0`. Build verification does not prove compatibility with a deployed kernel or installed firmware. The production driver pin remains unchanged. No module load, device ioctl, reset, or firmware update ran.

The upstream host-only firmware-log test passed all 109 checks. It covers valid records, malformed records, truncated headers, and text sanitization. The `make -C test check` target runs no hardware tests.

## TT-Metal 0.77.0: patch migration blocks the upgrade

A dry-run patch check against the 0.77.0 release archive found these concrete constraints:

| Local patch | Observation |
|---|---|
| `cadical-local.patch` | The patch tool reports a reversed or already-applied change. |
| `header-only-local.patch` | The patch tool reports reversed or already-applied changes. |
| `local-find-package.patch` | Applies with offsets and fuzz. This is not acceptance evidence. |
| `patched-pins-local.patch` | Applies with offsets. |
| `bh-eth-dispatch-harvesting.patch` | Applies with an offset. |
| `gdn-decode-fp32.patch` | Five integration hunks fail across CMake, source lists, and nanobind registration. |
| `umd-asio-local.patch` | Not evaluated: the release archive omits the UMD submodule contents. |

The GDN patch adds custom host and device code. A version-only replacement cannot preserve that integration. The remaining work requires a reviewed patch migration, the matched UMD submodule, and fresh kernel and model checks. This evaluation did not build or run TT-Metal 0.77.0.

The local Qwen model source remains `d102e7f4512060e09e580e40cf26a870bfa88c55` from `pkgs/qwen36/source.nix`. A newer runtime does not prove compatibility with that model tree.

## vLLM: standard plugin support is real

Reviewed source: [tenstorrent/vllm-tt-plugin at bef89e429e202caa38d3e3c8a24da4e8cc02405a](https://github.com/tenstorrent/vllm-tt-plugin/tree/bef89e429e202caa38d3e3c8a24da4e8cc02405a).

This plugin replaces the deprecated `tenstorrent/vllm` fork. Its current installer builds upstream vLLM 0.26.0 with `VLLM_TARGET_DEVICE=empty`. It installs the common dependencies separately to avoid the CUDA package set. TT-Metal owns the compatible PyTorch environment.

The source declares these entry points:

- `vllm.platform_plugins:tt`
- `vllm.general_plugins:tt_model_registry`

The source also declares `P150x2` as a `(1, 2)` mesh. The Qwen3.5 registration selects `models.demos.blackhole.qwen36.tt.qwen36_vllm:Qwen36ForCausalLM` by default.

These declarations establish a relevant integration path, not local serving evidence. Important limits remain:

- `platform.py` rejects speculative decoding and LoRA in its normal configuration path.
- Chunked prefill, prefix caching, and async decode require model capability declarations.
- The plugin targets the vLLM V1 model-runner contract.
- The README requires a model-matched TT-Metal/vLLM pair. A successful `ttnn` import is only platform detection, not a compatibility check.

The speculative-decoding rejection prevents direct substitution for this repository's MTP service. No vLLM installation, plugin host test, model load, or server execution ran. The plugin is a pinned reference, not an enabled package or service.

## Evidence and reproduction

The final [simulator receipt](evidence/tenstorrent-upgrade-20260907/ttsim-single-chip.json) records version 1.10.6 and status `passed`. Its BLAKE3 digest is `247cc45d23fa57accb2a96b9b323ef5d5bb25a5537c5749daae4aeb34e2710d5`.

Both focused Nix checks passed. The runtime-bundle output is `/nix/store/52brm8nfbm67xmqqczm4qayzngrl0v0f-ttsim-runtime-bundle-check`. The smoke output is `/nix/store/7l80i3zkpbi1gsy66ca19y832qv6hgcs-ttsim-blackhole-single-chip-smoke`.

Local command logs are under `~/.cache/tt-upgrade-review/evidence/`. The relevant files are `baseline-store-build-dir.log`, `baseline-smoke.log`, `validator-after.log`, `version-drift-negative.log`, `acceptance.log`, `kmd-candidate-build-final.log`, and `kmd-host-tests.log`.

The baseline Nix validator build passed 12 unit tests and 5 fixture tests. The updated validator passed the same suites, strict Clippy, and Rust formatting. A negative Nix evaluation with simulator version 1.10.2 failed the new version assertion.

Run the focused package and simulator checks with:

```console
nix build .#checks.x86_64-linux.ttsim-runtime-bundle \
  .#checks.x86_64-linux.ttsim-single-chip-smoke --no-link -L
```

The host's `/tmp` quota was full during this evaluation. The successful builds used a private `TMPDIR` and `--option build-dir /nix/var/nix/builds`. No unrelated temporary files or caches were removed.

These results do not establish physical placement, hardware performance, recurrent-state correctness, or production readiness for TT-Metal 0.77 or vLLM.
