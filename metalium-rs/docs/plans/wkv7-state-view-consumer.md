# Native F32 WKV7 state-view consumer

## Contract and selection

`TT_WKV7_STATE_VIEW=1` explicitly opts Burn's batched WKV7 recurrence into
consuming the existing state view of its preceding combined output. The default
remains the compact-state path. This is independent of `TT_DECODE_MATVEC`.

The real model path is `rwkv7/src/layer.rs::fused_tt_recurrence`: WKV7 returns
`[768,96]`, the state is columns `0..64`, reshaped to `[12,64,64]`, and the next
token's `burn_tt::wkv7_step_batched` previously calls `plain_dram` on this slice.
That materializes 48 F32 tiles, 196608 payload bytes, before the consuming kernel
reads those same 48 tiles. The new route recognizes this exact layout using the
existing `Strided` representation and retains its parent allocation. The consumer
reads tiles `(2*head + row_half)*3 + column_half`, rather than using a compact
pitch of two. It neither copies the state to GDDR nor downloads it.

This is a whole-tile F32 consumer, not BF16 affine-face-row gathering, a general
strided elementwise implementation, or a second view abstraction. It does not
remove the separate last-column recurrent-output materialization. Unsupported
layouts use the established native materialization path, not an alias pretending
to be supported. Only F32, origin `[0,0]`, parent `[768,96]`, strides
`[[64,0],[1,0],[0,1]]` are eligible. Compact state `[768,64]` remains supported.
The SFPU programs, operands, operation order, padding contracts, and parent bytes
are unchanged. Excluded columns cannot enter recurrence arithmetic.

## Evidence gate

From the repository root, simulator (no hardware):

```sh
nix develop .#metalium-rs --command cargo test --manifest-path metalium-rs/Cargo.toml \
  -p tt-tests --test step191_wkv7_state_view -- --nocapture
TT_WKV7_STATE_VIEW=1 nix develop .#metalium-rs --command cargo test \
  --manifest-path metalium-rs/Cargo.toml -p tt-tests --test step191_wkv7_state_view -- --nocapture
```

After the simulator passes, the hardware-window owner runs isolated card 0:

```sh
# Run from metalium-rs/.
TT_WKV7_STATE_VIEW=1 nix develop .#metalium-rs --command cargo xtask silicon \
  --release --device 0 --filter step191_wkv7_state_view --timeout-secs 180
```

Run the last command from `metalium-rs/` (Cargo aliases belong to that workspace).
Do not use plain `cargo test --features silicon`. Follow the repository's native
service-isolation/reset procedure before opening hardware.

The gate compares every output bit against a compact-state consumer, poisons
excluded parent columns with payload NaNs, checks live signed zero, subnormal,
NaN and infinity parity, checks the parent's original bits, and checks standalone
transfer packets do not increase during direct consumption. Burn routing carries
two tokens without downloads; a third transposed-state step must materialize,
proving that an unsupported layout is not silently accepted as an alias.
Timing uses two warmups and seven silicon repetitions (one simulated repetition),
host synchronization and validation of every timed output outside the timed
region. `WKV_STATE_VIEW` rows report the median microseconds, packet and compute
batch deltas, and direct/compact ratio. Simulated timing is explicitly labelled
and is not a performance claim. Transfer counters include warmups; timing does
not. Silicon qualification requires correct outputs, zero direct standalone
transfers, and a warmed nonregression before changing defaults. Full-model state
carry and end-to-end performance remain separate admission gates.

## Observed simulator evidence (2026-10-23)

The before-implementation gate failed at the direct consumer with
`Shape("wkv7 batched state: [768x96], expected [768x64]")`.
After implementation, both focused gates passed with `TT_WKV7_STATE_VIEW=1`:

```text
WKV_STATE_VIEW_BURN token=0 enabled=true state_materializations=0 downloads=0
WKV_STATE_VIEW_BURN token=1 enabled=true state_materializations=0 downloads=0
WKV_STATE_VIEW_BURN token=2 enabled=true state_materializations=1 downloads=0
WKV_STATE_VIEW simulated=true materialize=true reps=1 median_us=167274 transfer_packets=3 batches=36
WKV_STATE_VIEW simulated=true materialize=false reps=1 median_us=155502 transfer_packets=0 batches=36
```

The same compiled simulator binary passed with the opt-in absent: each of all
three routed steps materialized once, with no intermediate downloads. Timing
varied between simulator runs, including a direct/compact ratio above one;
none is physical performance evidence. Physical results are recorded below.


## Observed physical card 0 evidence (2026-10-23)

The isolated silicon runner executed both state-view gates in release mode,
with `TT_WKV7_STATE_VIEW=1`, physical card 0, tile `(3,4)`. Final run
`1792780405` passed, including exact alias/poison/special-value/stateful checks.
The two supported routed states submitted zero materializations and downloads;
the unsupported third transposed state materialized once, without a download.
Two warmups and seven timed samples measured compact-copy-plus-consumer median
**817 μs**, direct consumer **807 μs** (ratio **0.988343**). Direct standalone
transfer packets were zero versus nine compact packets including warmups; both
submitted 108 compute batches. These are bounded consumer measurements, not a
claim of improved complete-model throughput.

The existing 12-layer checkpoint oracle gate on card 0 with twelve worker tiles
passed with this flag alone and with `TT_DECODE_MATVEC=1`, on the fast route and
on the default accurate route. Both flags also passed the existing three-token
stateful generation gate. Exact commands, binary/checkpoint/oracle hashes, raw
model outputs and consumer medians are in the
[native qualification receipts](../../../.cairn/changes/native-decode-strip-matvec-view-consumers/evidence/decode-matvec.md#card-0-physical-qualification-2026-10-23).
The flag remains **off by default** pending broader model performance admission.

## Reference boundary

The pinned [ttPseudoRowMajor tensor view implementation](https://raw.githubusercontent.com/marty1885/ttPseudoRowMajor/5548f9d43ec9648de31a51c5e9d42599101b83ec/ttprm/kernel/tensor_view.hpp)
distinguishes whole-tile reads from affine face-row gathers, but its 32-byte rows
are BF16-specific. No BF16 address arithmetic is borrowed for this F32 path.
Native `wkv7.rs`, `READ_RUN` and existing streaming ownership are the executable
contracts. This is independent of the historically rejected bank-matched decode
candidate and does not revise that rejection.
