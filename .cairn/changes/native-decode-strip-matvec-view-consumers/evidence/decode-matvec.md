# Native decode live-row matvec evidence

## Mechanism and boundary

The opt-in `TT_DECODE_MATVEC=1` configures `Session::set_decode_matvec(true)` through the native Burn silicon attachment. It specializes ordinary, non-batched products whose logical M is exactly one. It retains the existing F32 tiled GDDR storage, transpose-aware gather, K-block continuation, fidelity phase order, semaphore ownership, zero-clear and full-tile pack. Other M values use the established matmul. Invalid environment values error at attachment. The default is off.

The new math issues only the first eight-row MVMUL group, rather than all four groups of a 32-row tile: only row zero is logically live. It avoids the three padding groups, including their activation/weight face unpacks, without changing any live output's product order. The output still has normal zero padding for skipped rows. This is **not** tt-lab's packed BF16/BFP8 strip storage or its three-stage/two-outstanding-read schedule; the native scheduler and existing double-buffer selection remain authoritative. No aggregate weight-byte reduction is claimed.

Inspiration was source-reviewed at `tenstorrent/tt-lab` commit `e6baf562a72491ed9d594cddbdf42e0925b1f79e`, `src/brisc.cpp:558-655`, including `mvmul_math64` and its read schedule. That Apache-2.0 source was not copied. Local original code remains LGPL-2.1-only. The MVMUL eight-row contract is documented in the pinned ISA's `WormholeB0/TensixTile/TensixCoprocessor/MVMUL.md`; existing native encoders carry the measured Blackhole bit positions.

## Exercised device-free proof

From `metalium-rs/`:

```console
nix develop .#metalium-rs --command cargo test -p tt-kernels --lib matmul::tests::each_phase_is_one_more_mvmul_per_half --no-default-features
nix develop .#metalium-rs --command cargo test -p tt-tests --test step192_decode_matvec -- --nocapture
```

First command: 1 passed, 197 filtered out. Second: 1 passed, 0 failed, 0 ignored, simulated execution 0.22 s. Toolchain is the package's Rust 1.97.1. The second gate opens the native TTsim 1.10.9 session and executes the actual new kernel. Coverage: independent exact sparse/integer oracle, logical `[1,65]` × `[65,35]`, physical normal/transposed right operand, all four fidelities, forced one-tile K continuation, and bitwise fractional baseline comparisons for TF32-from-F32 and BF16-from-F32 source routes. This is simulated numerical evidence, not silicon performance.

## Hardware ownership receipt

The installed `/run/current-system/sw/bin/ttwkv7-owner-control` is configured for `qwen38-p150x2.service` and device 1 (different from the checked-in package call's historical owner). Observed qwen38 active and old docker owner inactive. Used the installed allowlisted helper, not a service restart:

```console
ttwkv7-owner-control isolate && tt-smi -r
```

Exact terminal output:

```text
ttwkv7-owner-control: owner isolated
Resetting all PCI devices: [0, 1]
 Re-initializing boards after reset....
```

Elapsed 46.23 s. The service remains stopped. This receipt establishes isolation/reset only, not successful native execution.

## Card 0 physical qualification (2026-10-23)

After simulator success, the independent negative control changed the first
MVMUL row from zero to eight. The same simulator gate failed at element zero
(`0 vs -3`), then passed again after restoring the production row selection.

Final physical command, run from `metalium-rs/` with the owner isolated:

```sh
TT_WKV7_STATE_VIEW=1 nix develop .#metalium-rs --command cargo xtask silicon \
  --release --device 0 --filter step192_decode_matvec \
  --filter step191_wkv7_state_view --timeout-secs 300
```

Run `1792780405`: four tests passed, none failed. This was physical **card 0**,
not the simulator. Exact command, dimensions, medians and routed-copy counts are
in [kernel-card0-2026-10-23.json](kernel-card0-2026-10-23.json). Two warmups and
seven synchronized timed samples used sparse independently validated weights at
the actual checkpoint projection and LM-head dimensions:

| Logical M×K×N | Baseline μs | Candidate μs |
|---|---:|---:|
| 1×768×768 | 467.107 | 469.912 |
| 1×768×3072 | 1827.080 | 1827.271 |
| 1×3072×768 | 1684.692 | 1679.833 |
| 1×768×65536 | 33776.860 | 33806.856 |

These near-flat results do **not** qualify a performance promotion. The live-row
implementation is complete and numerically accepted; `TT_DECODE_MATVEC` remains
off by default. This is independent of the older rejected bank-matching work.

## Checkpoint, model oracle and continuation

Built the actual release binaries with
`nix develop .#metalium-rs --command cargo build -p rwkv7 --bins --release`.
[model-card0-2026-10-23.json](model-card0-2026-10-23.json) binds their SHA-256,
checkpoint revision `d81965cb4e1a9f96696b4f70b84212b8f2e43216`, checkpoint SHA-256,
independent Python oracle hashes, exact commands/environments and full outputs.
The checkpoint is RWKV7 Goose World2.8 0.1B, 12 layers, hidden 768, FFN 3072,
vocabulary 65536. The existing gate uses physical card 0, 12 worker tiles and the
fused native recurrence, not a synthetic layer substituted for the model.

The model-oracle command was `rwkv7-model-gate --checkpoint $PINNED_CHECKPOINT
--oracle $PINNED_ORACLE --card 0 --fused --tiles 12 --fast`, executed with neither
flag, only `TT_DECODE_MATVEC=1`, only `TT_WKV7_STATE_VIEW=1`, and both. All four
passed with identical reported hidden/logit/state error summaries and greedy
token 2. Fast-route gaps against the independent oracle were hidden `1.210e-1`,
logits `3.928e-2`, states `3.027e-2`, within the pre-existing gate bounds.

Both flags also passed the same command **without `--fast`**: oracle gaps hidden
`1.619e-3`, logits `2.756e-4`, recurrent states `4.101e-4`. Finally,
`TT_DECODE_MATVEC=1 TT_WKV7_STATE_VIEW=1 rwkv7-generate-gate --checkpoint
$PINNED_CHECKPOINT --card 0 --fused --tiles 12 --fast` passed three stateful decode
steps, selecting token 1 each time against its independent expected tokens.
These are bounded numerical/continuation gates, not a full perplexity screen or
warmed end-to-end throughput admission. Both defaults remain unchanged.
