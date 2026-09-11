# Persistent decode window acceptance

## Result

The RWKV-7 P150x2 runtime now admits decode windows of two and four tokens.
Unsupported windows use normal dispatch and report `window_not_admitted`.

## Correctness

The longer-window test used the tokenizer-backed `Hello` prompt.
Its token IDs were `[0, 24281, 59, 36786, 65530, 5585, 41693, 59]`.
The pinned FP32 continuation was `[36786, 34, 308, 459]`.

The production receipt reports exact generated-token parity.
It also reports exact FP32 recurrent-state parity.
All four state slots were acknowledged, zeroed, and released.

The two-token window passed again under profile `rwkv7-p150x2-persistent-decode-v2`.
The one-token physical request selected normal dispatch.

## Simulator

The Blackhole simulator executed a four-step, one-layer Metal Trace replay.
The trace used one host replay submission and device-resident FP32 recurrent state.
The simulator receipt makes no performance or determinism claim.

## Physical A/B evidence

Five order-balanced pairs ran on one P150 card.
Five order-balanced pairs also ran on both P150 cards with synchronized timed starts.
The two-card worker intervals overlapped in every retained wave.

The one-card minimum paired speedup was `2.849279940121875x`.
The one-card median paired speedup was `2.867805089379262x`.

The two-card minimum paired speedup was `3.011763122817449x`.
The two-card median paired speedup was `3.084333977350274x`.

These ratios compare equal-workload elapsed times.
They do not use the generic throughput token counter as the numerator.

## Profiler

The final production run used trace-only profiling without a candidate override.
The retained CSV contained 520 data rows and no dropped-marker warning.
Its BLAKE3 identity is `9fadc80527a543365b287e2a6f3479299863f981fd6b468465709a2fd411f14c`.

## Terminal cleanup

Physical pre-replay cancellation, timeout, and failure probes passed.
Each probe acknowledged the terminal event, zeroed device state, and released all four slots.
Five persistent A/B samples reused slot IDs 0 through 3 after earlier worker cleanup.

## Retained evidence

- `physical-acceptance.json`
- `production-profiled-worker-receipt.json`
- `simulator-receipt.json`
- `terminal-cancel-receipt.json`
- `terminal-timeout-receipt.json`
- `terminal-failure-receipt.json`
- `fallback-receipt.json`
- `short-window-receipt.json`

`physical-acceptance.json` passed the strict Rust receipt validator.
The production worker receipt also passed the production policy.
