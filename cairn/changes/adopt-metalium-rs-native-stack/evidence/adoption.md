# metalium-rs Adoption Evidence

Date: 2026-10-19

## Fork identity

- Fork: `https://github.com/brittonr/metalium-rs`
- Commit: `d8c6e2fc4579cf29aa7525ddd086a90f81da2260`
- Upstream base: `Knight-Ops/metalium-rs` at `b22a245db18252b131679aca992dac85c9f7d812`
- Fork tree BLAKE3: `2a424a20cf34fdccbc68071d00d5eae8cf0bc8bae9d03cc79c660ac57479bb00`
- `Cargo.lock` SHA-256: `13231685f3a42f549454864e696af3d4a3c35f7d54113b94fbcbf9d64ce57501`
- Nix source hash: `sha256-izG12OWV37sIUiQFYam701M6FR5kUAvqYvvLeEEqT28=`
- Fork delta: `LICENSE` (Apache-2.0 text), `crates/burn-tt/tests/essentials.rs` (F16 capability
  assertion now expects `Storage | Arithmetic`, without `Accelerated`), and `crates/rwkv7`
  (WKV reference/device gate and synthetic full-layer gate). Nine files, +1,146/−2.

## Validation

`cargo test --workspace --no-fail-fast` at the fork commit:

- 1,542 passed
- 0 failed
- 46 ignored
- 263 test binaries

The run used the pinned ISA specification, ttsim `v1.10.9`, and MNIST inputs. Simulator gates are
simulated evidence. No physical device was opened.

## Package

- Flake input: `rust-overlay` at `550547cefc67e01a6fd11a09e4096a3c844ca840`, following the
  repository nixpkgs.
- Toolchain: Rust 1.97.1 with `riscv32im-unknown-none-elf` and `llvm-tools`.
- Package: `.#metalium-rs`, revision `8600f14`, `Cargo.lock` from the source.
- MNIST: four CVDF files fetched with the upstream `PINS.toml` SHA-256 values and decompressed.
- Excluded dev-only crates: `tt-tests`, `tt-ttsim`, `tt-ttsim-sys`, `burn-tt-conformance`,
  `tt-models`, `xtask`.
- Output: `/nix/store/qvs6jxcbw6dq91plm73a0ykp64asdnkv-metalium-rs-0.1.0-unstable-2026-10-19`
- Installed artifacts: `bin/tt-mnist` (26 MiB), `bin/wkv7-gate` (4.4 MiB), and
  `bin/rwkv7-layer-gate` (6.1 MiB).
- The package check phase runs `cargo test --workspace --no-fail-fast` with the pinned specification,
  ttsim `v1.10.9`, MNIST, `cfg_defines.h`, and `ioctl.h`. It passes: 1,542 tests, 0 failures,
  46 ignored, 263 binaries, in about 13 minutes.
- App: `.#metalium-rs-mnist`.

## Binary smoke

- `tt-mnist --help` prints the usage text and exits 2.
- `tt-mnist --infer --host --batch 1 --passes 1` loads the embedded corpus
  (60,000 train / 10,000 test images), prints the model and storage plan, then fails at
  `could not open the card: Invalid argument (os error 22)`. The host flag does not bypass the
  device open in this mode.

The smoke proves the binary starts, parses arguments, and unpacks the embedded corpus. It was run
before isolation, when `qwen38-p150x2.service` held the cards, so it does not prove a successful
device path.

## Physical gate

`qwen38-p150x2.service` was stopped through the permitted `systemctl stop` rule before these runs.
The service was inactive and `lsof` showed no owner on `/dev/tenstorrent/1`.

Both runs exited 0:

| Command | Result |
|---|---|
| `nix run .#metalium-rs-mnist -- --steps 1 --card 0` | loss 2.3095; 127.34 ms/step; test accuracy 13.14%; 221.6 MB up, 0.04 MB down |
| `nix run .#metalium-rs-mnist -- --steps 1 --cards 0,1` | loss 2.3095; 119.18 ms/step; test accuracy 13.14%; 221.6 MB up, 0.04 MB down |

Both report a 784-512-256-10 network and one Tensix tile per invoked card. The mesh run reports
matmuls sharded over Ethernet. These are single-step smoke timings, not benchmarks: no warmup,
median, or repeated-run protocol was applied.

## WKV7 native gate

The fork now carries `crates/rwkv7`: the token-at-a-time Goose recurrence as a pure F32 host
reference and the same equation composed from Burn tensor operations on `TtBackend`. It uses
`accurate_matmul`, not the fused Tensix kernel.

`cargo run -p rwkv7 --bin wkv7-gate -- --card 0` and `nix run .#metalium-rs-wkv7 -- --card 0`
on 2026-10-19:

```text
host reference: step1 out 0.406937, state 0.247399; step2 out 0.236715; control gap 0.311922
card 0: step1 out gap 1.192e-7, state gap 2.980e-8; step2 out gap 5.960e-8, state gap 2.980e-8
device gate PASS
```

The second token consumes the first token's device state, so this is a state-carry check. It is one
card, two tokens, and a composed op graph. The packaged binary reproduced the same gaps. It does not establish a fused-kernel cost, a layer, a
model, or mesh behavior.

## Synthetic RWKV-7 layer gate

`crates/rwkv7/src/layer.rs` implements one layer from the pinned checkpoint equations: pre/attn/ffn
norms, token shift, the time-mixing LoRA projections, the WKV recurrence, group norm, value
correction, output projection, and the squared-ReLU channel mixer. The gate runs layer zero with
synthetic weights on `Flex` (CPU FP32) and `TtBackend`, for two tokens, with `accurate_matmul`.

`nix run .#metalium-rs-rwkv7-layer -- --card 0`:

```text
flex: token1 max 1.834199, token2 max 1.805208, state max 0.156451; control gap 0.000691
card 0: token1 hidden gap 5.960e-7
card 0: token2 hidden gap 3.576e-7
card 0: token1 state gap 1.788e-7
card 0: token2 state gap 1.490e-7
layer gate PASS
```

Two fresh runs produced the same gaps. A first attempt immediately after stopping the service
failed with a stopped backend queue (`element-wise 0x2 ... queued list 2 failed`); the cause is not
established and the later runs were clean.

The gate uses synthetic weights, layer zero only (no v-LoRA), and no real checkpoint. It is not a
real-weight layer, full-model, fused-kernel, or performance result.

## Evidence classes

- Device-free: fork build, package build, hermetic Nix test lane, `--help`.
- Simulated: the ttsim gates inside the fork validation tier.
- Physical: one training step on card 0, one on the `1x2` mesh, a two-token WKV7 device gate, and
  the synthetic full-layer gate on card 0. No real-weight model inference, placement, long-run
  stability, or performance claim follows.

## Migration status

The native stack is the authority for new Rust device, firmware, kernel, and Burn-backend work.
Qwen3.6/3.8 and RWKV-7 remain on their TT-Metal/ttnn paths. Their native migration is blocked on
operator coverage, linear-attention/MoE kernels, recurrent-state carry, and physical evidence.
The per-target inventory is `docs/research/metalium-rs-parity-inventory.md`.
