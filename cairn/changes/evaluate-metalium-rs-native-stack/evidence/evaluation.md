# metalium-rs Evaluation Evidence

Date: 2026-10-19

## Source identity

- Repository: `https://github.com/Knight-Ops/metalium-rs`
- Commit: `b22a245db18252b131679aca992dac85c9f7d812` (2026-10-14)
- Tracked-tree BLAKE3: `4baeae7cc6c8d8a733f73805153a6bf63c47238958ba73336f669f51d8b9ddaa`
- `Cargo.lock` SHA-256: `8c974b0aa5270fbe13dc463ee08333b440d8454c40f60b504d000bf3fc79801a`
- License: `license = "Apache-2.0"` in the workspace manifest. No `LICENSE`, `COPYING`, or `NOTICE` file is tracked at this commit.
- Checkout: `.pi/references/tenstorrent/metalium-rs` (not version-controlled)
- Source modifications: none. `git status --porcelain` is empty.

## Host and toolchain

- Host: Linux 6.18.50, x86_64. Two `/dev/tenstorrent` nodes; tt-kmd `2.10.0`.
- Pinned tt-kmd ABI: `ttkmd-2.11.0`, IOCTL API version 2 (`crates/tt-kmd/src/lib.rs:43`). The host's 2.10.0 header differs only by the additive `SMC_MSG` ioctl and structure; the tree does not call it.
- Nix development shell: Rust 1.95.0. No `riscv32im-unknown-none-elf` standard library and no `llvm-objcopy` or `llvm-objdump`.
- Evaluation toolchain: rustup stable 1.97.1 (`8bab26f4f`) with `rust-std-riscv32im-unknown-none-elf` and `llvm-tools` added.
- Environment repair: the rustup `gcc-ld/ld.lld` wrapper referenced a garbage-collected `/nix/store/...-rustup-1.29.0/nix-support/ld-wrapper.sh`, so host links failed with exit 127. The wrapper was replaced with an `exec` of `gcc-ld-unwrapped/ld.lld`. This is environment state, not a source change.
- Storage: `/tmp` was full; `CARGO_TARGET_DIR` and `TMPDIR` were moved to `~/.cache`.

## Pinned inputs fetched

- ISA specification: `f848eb668c2aeae742a88a49a86157e24a0a20c6`
- ttsim: `v1.10.9` (`libttsim_bh.so`, `libttsim_bh_x2.so`, `libttsim_bh_x4.so`)
- MNIST: the four CVDF files with their published digests
- tt-metal `cfg_defines.h`: `81989dcdb8f9b340c932ae7a71a346f4f08703eb`
- Burn: `=0.21.0` from `Cargo.lock`

## Results

| Check | Result |
|---|---|
| `cargo fmt --all --check` | pass |
| `cargo clippy --workspace --all-targets -- -D warnings` | pass |
| `cargo xtask check-no-sim-in-ship` | pass |
| `cargo xtask check-no-flex-in-backend` | pass |
| `cargo xtask gen-cfg --check` | pass |
| `cargo xtask gen-isa --check` | pass |
| `cargo xtask gen-burn-ops --check` | pass |
| `cargo xtask burn-coverage --check` | pass |
| `cargo test --workspace --no-fail-fast` | 1,538 passed, 2 failed, 46 ignored across 259 test binaries |

The first run failed before test execution for one target because `vendor/cfg_defines.h` was absent. `cargo xtask gen-cfg` fetched it, and `cargo test -p tt-tests --test step120_tag_search` then passed 8 of 8.

## Remaining blocker

`cargo test -p burn-tt --test essentials` fails one test at the pinned commit:

- Test: `dtype_capabilities_advertise_what_the_device_computes_and_no_more`
- Failure: `crates/burn-tt/tests/essentials.rs:23`
- Assertion: F16 must contain `DTypeUsage::Storage` and must not contain `Arithmetic` or `Accelerated`.
- Implementation: `crates/burn-tt/src/lib.rs:217` returns `DTypeUsage::general()` for `DType::F16`.
- Introducing change: `cff6742` ("F16 arithmetic, and Burn's F16 suites in every conformance profile") added F16 arithmetic after the test was written in `07fcae0`.

The test and implementation disagree at the same commit. No source patch was applied.

## Evidence classes

- Device-free: formatting, clippy, generated-file checks, boundary checks, and non-simulator tests.
- Simulated: tests that load `libttsim` `v1.10.9`.
- Physical: none. No `/dev/tenstorrent` node was opened, no firmware was loaded onto hardware, and no model inference, placement, performance, or silicon conformance was measured.

## Verdict

`blocked` for package or runtime authority. The pinned revision builds and passes the generated-file and boundary checks, but it does not pass its own validation tier, and the nixpkgs toolchain lacks the firmware target and LLVM tools. Reference authority only.

## Fork follow-up

The repository forked the source to `brittonr/metalium-rs` and pinned commit
`5214c3d0c1a1bbf21f49020fc76e26d932f0f19d`. The fork adds the Apache-2.0 license text and changes
the stale F16 capability assertion to `Storage | Arithmetic`, without `Accelerated`, matching
`DTypeUsage::general()`.

At the fork commit, `cargo test --workspace --no-fail-fast` passes 1,540 tests with 0 failures and
46 ignored tests across 259 test binaries. The Nix package in `pkgs/metalium-rs` builds the shipped
stack and the `tt-mnist` binary with a pinned Rust 1.97.1 toolchain. See
`cairn/changes/adopt-metalium-rs-native-stack/evidence/adoption.md`.
