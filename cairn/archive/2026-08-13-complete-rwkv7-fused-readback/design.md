# Design: Complete the RWKV-7 fused WKV readback

## Goal and completion evidence

The goal is a bounded physical receipt from the diagnostic BF16 readout runtime. It must prove exact FP32 state and BF16 row parity against package-matched generic TT-NN.

Completion requires:

- the exact diagnostic runtime and kernel root;
- finite diagnostic launches and synchronized readbacks;
- exact comparison of all FP32 state values and all raw and corrected BF16 bits;
- generated JIT evidence for recurrence and readout fidelity;
- unowned devices after every physical run; and
- no production admission change before a separate full evaluation passes.

False completion includes treating asynchronous enqueue as execution, accepting a missing receipt, weakening exact comparison, using symmetric fixtures, or starting admission gates after failed parity.

## Bounded portfolio search

Subagent consent is absent. These serial lenses are correlated.

| Family | Mechanism | Discriminating check | Result |
|---|---|---|---|
| Recurrence precision | FP32 values can be reduced by a non-exact unpack path or a different operation class. | Compare each recurrence boundary with generic TT-NN. | Exact unpack and SFPU operation classes were required. |
| BF16 readout contract | Fidelity, source order, transpose, accumulator width, geometry, or packing can change output bits. | Run a separate readout program and compare raw BF16 bits. | Reverse order, in-matmul transpose, HiFi2, and 16-bit accumulation passed. |
| Correction geometry | Raw `[1,12,64,1]` rows can broadcast incorrectly with `[1,12,1,64]` correction inputs. | Check the corrected element count and exact bits. | Canonical row reshape fixed the `49,152`-value broadcast. |
| Completion ownership | Compute and writer can race when they share one CB acknowledgement. | Audit each CB consumer and writer completion signal. | Dedicated output CBs and balanced signals passed. |

## Functional boundary

Pure Rust contracts validate source precision, ownership, geometry, and bounded diagnostic rules. The runtime shell launches finite programs, synchronizes readback, compares values, writes one receipt, and releases the device.

## Decisions

### Decision: Use a two-program diagnostic split

**Choice:** Run exact HiFi4 recurrence first. Run the proven HiFi2 BF16 readout in a separate program.

**Rationale:** One worker program cannot assign two compute fidelities on the same core.

### Decision: Preserve generic operation and materialization contracts

**Choice:** Use exact FP32 unpack, SFPU multiply for decay, SFPU add for recurrence sums, and the generic TT-NN materialization order.

**Rationale:** Arithmetic equations alone did not reproduce package-matched device bits.

### Decision: Give the caller correction ownership

**Choice:** Split mode returns raw readout with `correction_complete = false`. The caller reshapes rows and applies correction once.

**Rationale:** This makes ownership explicit and prevents duplicate correction.

### Decision: Preserve production behavior

**Choice:** Split mode requires both diagnostic precision flags. Production keeps the accepted profile and windows `[2, 4, 8]`.

**Rationale:** Diagnostic completion is not a window-16 admission result.

## Physical budget

- Device: physical device `0` for the bounded diagnostic probe.
- Each attempt has a named timeout and a new `/var/tmp` root.
- No unbounded loops or automatic retries.
- Both devices must be unowned after each attempt.
- A failed probe cannot start profiler, A/B, canary, monitoring, deployment, or admission work.

## Risks / Trade-offs

- The scalar host recurrence is not package-bit-exact. Generic TT-NN remains the exact device authority for this diagnostic.
- The package-matched Blackhole corpus does not document every NoC detail. The safety verdict relies only on explicit barriers.
- The split path is diagnostic-only. A new immutable window-16 candidate must pass every later gate.
