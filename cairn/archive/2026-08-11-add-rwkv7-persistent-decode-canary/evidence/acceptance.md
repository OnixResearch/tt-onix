# Persistent decode canary acceptance

Date: 2026-08-11

## Result

The bounded physical canary passed on P150 devices `0` and `1`. The final runtime interlock state is `retain`.

The canary completed in `575.768298028` seconds. The policy limit is `600` seconds.

The controller planned `72` request executions under a hard maximum of `104`. The candidate share was `0.5`.

## Matched physical A/B

The canary used four matched samples. One-card worker pairs used separate cards and swapped card assignments after each sample.

Dual-card control and candidate samples alternated order. Both workers used synchronized timed starts.

- Minimum one-card paired speedup: `3.0575010792776935x`.
- Minimum two-card paired speedup: `3.145531054606447x`.
- Exact generated-token parity: passed.
- Stable positive gain: passed.
- Physical A/B receipt BLAKE3: `f89a72fe330107ebed2def604ec109ee1bce7c87f1a9432a5b5c3d0fc8266cbb`.

## Candidate monitoring

The controller collected only candidate worker telemetry from known receipt paths.

- Accepted events: `12`.
- Rejected events: `0`.
- Persistent trace events: `12`.
- Completed requests: `32`.
- Token and FP32 state parity events: `12` passed.
- Complete cleanup events: `12`.
- Fallbacks: `0`.
- Timeouts: `0`.
- Terminal failures: `0`.
- Alerts: `0`.
- Verdict: `clean`.
- Monitoring batch BLAKE3: `be42e25c751c29615accf1e71b5d5c6a004de7b4905504989db616418854e60c`.

## Runtime interlock

Every canary worker received the same armed interlock path, exact file BLAKE3, and canary binding.

The final interlock has state `retain` and reason `clean_canary`. Its file BLAKE3 is `64fe04db1801fa22922c00890bb9d262736111e78b1bf75a704204108a084c41`.

Held, disabled, malformed, or mismatched interlocks force `normal_dispatch` with `canary_interlock_blocked`.

## Fail-closed attempts

A parser mismatch stopped the first attempt before timing. A later serial attempt reached the exact wall limit.

Both attempts wrote `disabled` interlocks. Neither attempt changed the production profile.

The accepted run used per-card TT-Metal caches and concurrent one-card worker pairs. The controller removed temporary caches before final evidence.

## Retained evidence

- `physical-ab-receipt.json`.
- `candidate-monitoring-receipt.json`.
- `canary-interlock.json`.
- `canary-receipt.json`.

## Non-claims

This canary does not route live service traffic. It does not mutate the installed profile or admit decode window `8`.

It does not provide network export, paging, or worker process supervision.
