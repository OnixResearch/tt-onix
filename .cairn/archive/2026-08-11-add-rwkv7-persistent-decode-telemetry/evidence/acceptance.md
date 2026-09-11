# Persistent decode telemetry acceptance

## Result

The receipt-native telemetry contract passed device-free and physical validation.

The runtime emits one `persistent_decode_telemetry` event with schema version `1`. The event contains only bounded categories, counts, and monotonic durations.

## Physical persistent execution

The profile-owned two-token path ran on physical device `0` without a candidate override.

- Selected mode: `persistent_trace`.
- Selected window: `2`.
- Completed slots: `4`.
- Generated-token parity: `passed`.
- FP32 recurrent-state parity: `passed`.
- Cleanup status: `complete`.
- Trace setup: `0.066039897` seconds.
- Trace replay: `0.011725462` seconds.
- Cleanup: `0.001032266` seconds.
- Receipt BLAKE3: `0d34ea64579e1e69e6c4b967374dff9085b696c1d4d697e4c8e4a79cc8e6bbac`.

The strict telemetry validator and the persistent candidate validator passed this receipt.

## Physical fail-closed fallback

An unlisted one-token request ran on physical device `0`.

- Selected mode: `normal_dispatch`.
- Selected window: `null`.
- Fallback reason: `window_not_admitted`.
- Terminal outcome counts: all zero.
- Parity status: `not_applicable`.
- Cleanup status: `not_applicable`.
- Trace setup, replay, and cleanup durations: zero.
- Receipt BLAKE3: `a4362c6c2b97721e6fd1744fffc995d4456f264e39cee9b6bcaa9d1775c307c4`.

The strict telemetry validator passed this receipt.

## Physical pre-replay cancellation

A four-token candidate received cancellation before trace replay on physical device `0`.

- Selected mode: `persistent_trace`.
- Selected window: `4`.
- Cancelled slots: `4`.
- Trace replay: `0.0` seconds.
- Parity status: `not_applicable`.
- Cleanup status: `complete`.
- Trace setup: `0.137646253` seconds.
- Cleanup: `0.003837858` seconds.
- Receipt BLAKE3: `23c2c6f9ed7e0b525f5ce208799cb2e5fed31a2c59c28b97f2794a06420fbb6b`.

The strict telemetry validator and the terminal-cancellation validator passed this receipt.

## Negative validation

Rust tests reject these cases:

- Missing telemetry.
- Unknown or high-cardinality telemetry fields.
- A dispatch or fallback mismatch.
- A terminal-count mismatch or duplicate lifecycle slot.
- A negative or non-finite duration.
- A parity claim without trace replay.
- A cleanup claim without released lifecycle slots.

## Non-claims

This evidence does not add a metrics transport, alert service, canary controller, or automatic admission disable action.

A fatal worker error can stop receipt creation. Process supervision remains responsible for that error path.
