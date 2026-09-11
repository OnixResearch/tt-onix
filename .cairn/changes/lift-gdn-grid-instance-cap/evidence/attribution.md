# Attribution: both GDN grid guards, their binding widths, and the B=8 rejection

Recorded 2026-09-11 for change `lift-gdn-grid-instance-cap`, tasks A1-A4.

## A1: both guards confirmed in the pinned source

**Decode.** The guard ships in the vendored fused-op patch. In the pre-change
`pkgs/tt-metal/gdn-decode-fp32.patch` (commit `8bf9f35^`) the hunk adds, at
patch line 324:

```c
TT_FATAL(num_heads <= compute_grid.x * grid_y, "num_heads exceeds the available compute grid");
```

Applied to the pinned source this lands in
`ttnn/cpp/ttnn/operations/transformer/gated_delta_decode/device/gated_delta_decode_program_factory.cpp:32`,
and the very next statement maps head `h` to core `{h / grid_y, h % grid_y}`:
one head, one core, no chunking.

Binary confirmation: `grep -l "exceeds the available compute grid"` matches
`lib/_ttnncpp.so` in the shared 0.77 runtime
(`/nix/store/dr9bj8qw…-tt-metal-0.77.0`) and matches nothing in the variant
runtime (`/nix/store/6ccynlvv…-tt-metal-0.77.0`), which is the change itself.

**Prefill.** In the pinned source
`/nix/store/5cafvl5c…-source/ttnn/cpp/ttnn/operations/transformer/chunk_gated_delta_rule/device/chunk_gdn_phased_program_factory.cpp`,
`distribute_scan` at line 137:

```c
const uint32_t ncores = grid.x * grid.y;
TT_FATAL(BH <= ncores, "num_heads {} exceeds compute cores {}", BH, ncores);
```

and the second cap at line 143 inside the `NV` search:
`if (Vt % cand == 0 && BH * cand <= ncores)`.

## A2: instance counts at TP=2 and where each guard binds

At TP=2 the fused decode op reports `H = 24` per device, so the per-device
instance count is `24 x B`.

**Measured compute grid, device-free.** On the packaged TTsim Blackhole `1x1`
runtime (variant `6ccynlvv…`, ttsim `zw3k9hs9…`, `TT_METAL_DISABLE_SFPLOADMACRO=1`)
the measured `compute_with_storage_grid_size()` is **13 x 10 = 130 cores**
(observation `compute_grid_measured`, exit 0).

| Batch | Instances (24xB) | vs 130 sim cores | vs physical cores |
| ----: | ---------------: | ---------------- | ----------------- |
| 1 | 24 | fits | fits (B=4 silicon admission proves >= 96) |
| 4 | 96 | fits | fits (fresh B=4 fused admission passed) |
| 8 | 192 | **exceeds** | **exceeds** (the B=8 fused path never ran) |
| 16 | 384 | **exceeds** | **exceeds** |

The exact physical harvested core count is measured during the V8 hardware
session; it does not change where the guard first binds, because both 110
(the planning constant used by the device-free probe) and 130 sit between 96
and 192. B=8 is the first width where both guards bind on either grid.

## A3: the decode guard reproduced device-free at B=8

Script `/tmp/gdn-grid-sim/overflow-only.py` under
`/tmp/gdn-grid-sim/run-guard-repro.sh` on TTsim Blackhole `1x1`:

**Shared (pre-change) runtime, 192 heads (B=8 at TP=2), 130 sim cores.**
The op call raises a catchable RuntimeError with the exact diagnostic:

```
TT_FATAL @ /build/source/ttnn/cpp/ttnn/operations/transformer/gated_delta_decode/device/gated_delta_decode_program_factory.cpp:32: num_heads <= compute_grid.x * grid_y
info: num_heads exceeds the available compute grid
```

`GDN_GUARD_SIM overflow_call flat_heads=192 outcome=raised`, clean close,
exit 0.

**Variant runtime, same call.** The factory accepts the 192-head launch
(`outcome: accepted`): the guard is lifted and the program is created with
the chunked assignment (ceil(192/130) = 2 per core, 96 active cores).

**The fallback that hides the guard in production.** The fused path is gated
by `QWEN35_GDN_DECODE_FUSION`, and `decode_fusion_candidate.selected_mode`
silently maps a missing or unknown value to `off`. No try/except wraps the
fused op call in `ttnn_delta_rule_ops.py` or `gdn/tp.py`; the fatal would
propagate. The hiding is one level up: because `full` was structurally
impossible at B=8 under the guard, the B=8 candidate was measured with the
fused path off, and no receipt field names the fusion mode. The guard
therefore surfaced as a performance near miss, not as an error.

**Simulator limitation recorded.** TTsim cannot execute this op to completion
at any head count: its tensix matmul model rejects
`ALU_ACC_CTRL_Zero_Flag_disabled_src=1 (keep SrcB denormals) not modeled`,
which aborts the simulator process during deferred execution. Chunking
execution and parity therefore require hardware; the simulator can prove only
launch-level behavior (which the two runs above do).

## A4: one guard family explains both missed B=8 bounds

The rejected B=8 receipt
(`pkgs/qwen36/evidence/2026-08-26/qwen38-27b-batch-admission-b8-receipt.json`)
failed three performance bounds:
`aggregate-decode-speedup-below-bound` (5.8815 vs 6.0),
`per-user-decode-retention-below-bound` (0.7352 vs 0.75), and
`ttft-ratio-above-bound` (9.0792 vs 9.0).

The archived physical result names the mechanism: "eight serialized per-user
prefills dominate each request". The receipt records
`batch/median_prefill_state_install_seconds = 1.3225` against
`single/median_prefill_state_install_seconds = 0.1257` (10.5x).

One cause explains all three misses: at B=8 the per-device instance count is
192, above the core count, so

- the **prefill** `distribute_scan` fatal refuses a batched phased prefill
  (`BH = 192 > ncores`), forcing the serialized per-user prefills that drove
  the TTFT ratio over its bound; and
- the **decode** factory fatal makes the fused GDN decode impossible at
  B=8, so decode ran unfused, costing the per-user retention and the
  aggregate speedup margin.

Both guards have the identical shape `instances <= cores`. One mechanism —
contiguous per-core instance chunking — lifts both at once. That is exactly
what this change implements, and it is why the readmission change re-runs the
unchanged bounds rather than editing them.
