# Design: Unblock the window-16 parity diagnostic

## Goal and completion evidence

The immediate goal is one bounded physical receipt from the diagnostic-only fused WKV runtime. The receipt must prove exact FP32 recurrent-state parity and exact BF16 corrected-row parity against a pure host reference.

Validated completion requires:

- the launched executable and kernel root belong to `rwkv7-p150x2-window16-readout-diagnostic-runtime`;
- the BF16 readout compile setting is true;
- no generic TT-NN operation runs before the fused program;
- the fused program compiles, executes, and writes a receipt within the named bound;
- FP32 state and BF16 corrected rows match exactly;
- both Tenstorrent devices are unowned after the run.

A mismatch or bounded timeout is a valid blocker. It must stop all later window-16 gates and preserve production windows `[2, 4, 8]`.

False completion includes running the rejected candidate, accepting a missing receipt, comparing only summaries, weakening exact parity, or starting admission gates after a failed diagnostic.

## Bounded portfolio search

Subagent consent is absent. These serial lenses are correlated.

| Family | Mechanism | Discriminating evidence | State |
|---|---|---|---|
| Generic device pre-work | The probe builds generic recurrence and correction operations before fused execution. | Watcher names show generic typecast, matmul, binary, layernorm, reduction, and reshape kernels before fused kernels. | Validated cause of excess work. |
| Fused JIT cost | The three fused kernels alone exceed the bound. | A fused-only probe with stage markers still times out in fused compilation. | Rejected. All three fused kernels compiled before synchronized readback stalled. |
| Cache conflict | Shared `TT_METAL_CACHE` causes lock or stale-build delay. | A unique explicit cache changes the terminal stage or duration. | Rejected as the first-attempt blocker. A unique cache reached fused launch completion. |
| Package binding | The launcher executes a non-diagnostic output. | The retained symlink resolves to the rejected candidate runtime. | Validated defect and corrected. |
| Device setup | Device initialization consumes the budget. | The `device-opened` marker is absent or late in the bounded log. | Rejected for the first attempt. The Ethernet-dispatch retry failed during firmware initialization. |
| Worker ownership | The rectangular compute grid includes dispatch-owned cores. | Compare the active logical cores with the package-matched Blackhole descriptor and TT-Metal grid construction. | Rejected for fast dispatch. The compute grid ends before the relative last-column dispatch cores. |
| Post-launch completion | The fused program does not reach synchronized readback completion. | `fused-launch-finished` is present, but `readback-finished` and the receipt are absent at timeout. | Retained blocker. |

## Functional boundary

The pure core receives host vectors and state. It computes the accepted recurrence, applies the generic BF16 readout boundary, performs group normalization and correction, and returns expected FP32 state and BF16 rows.

The imperative shell creates device tensors, launches only the fused operator, reads the two result tensors, compares them with the pure outputs, writes the receipt, and releases the device.

## Physical budget

- Device: physical device `0` only.
- Attempts: one fused-only diagnostic, with one retry only for a confirmed launcher or harness defect.
- Timeout: 30 minutes for the first fused-only run.
- Cache: one unique `/var/tmp` directory bound to `TT_METAL_CACHE`.
- Outputs: one bounded receipt plus aggregate stage and kernel-count evidence.

## Admission rule

Exact diagnostic parity permits a separate new immutable window-16 evaluation. That later evaluation must repeat CPU, simulator, full-model physical correctness, profiler, A/B, canary, monitoring, and deployment gates. This change itself does not admit window `16`.
