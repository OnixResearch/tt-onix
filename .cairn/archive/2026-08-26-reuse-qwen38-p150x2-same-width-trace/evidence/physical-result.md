# Physical result

## Verdict

The same-width trace-reuse candidate is **REJECTED**. Production keeps the admitted recapture path, and `physically_admitted` remains `false` in the trace-reuse binding.

- Receipt BLAKE3: `eff83e1a71548be95756691ef6434adfa1f6b053e2891209559033545d89dd22`
- Tracked receipt: `pkgs/qwen36/evidence/2026-08-26/qwen38-27b-trace-reuse-receipt.json`
- Candidate package: `/nix/store/8k8gwv01bbksc6861pn7x5wjrfsswana-qwen36-0.1.0`
- Execution identity: `/nix/store/cv0ana4k28b9v3cy81ky2yvmg23g1bil-qwen36-execution-identity-0.1.0`
- Runtime: `/nix/store/mg0ha5nsvnir1y1ysiyyzmm4hk83z3m6-tt-metal-0.74.0`
- Physical devices: `0` and `1`
- Root order: `recapture,reuse,reuse,recapture`

## What the mechanism achieved

Every measured reuse group served exact token and text parity with zero recaptures:

- exact token parity: pass (all four roots)
- exact text parity: pass
- one resident trace and zero measured recaptures: pass
- stable trace binding across prefill: pass
- median reused trace prepare: `0.001110980927478522` seconds (bound `0.05`)
- decode-rate retention: `1.0024661519899465` (bound `0.95`)
- tail-latency ratio: `0.8358347684174199` (bound `1.10`)

The recapture route kept its expected profile: median trace prepare `0.8465498390141875` seconds and one capture per group.

## Why it was rejected

Prefill under a resident trace is slower by more than the capture it avoids:

- reuse median TTFT: `5.608782936993521` seconds
- recapture median TTFT: `4.341617294005118` seconds
- TTFT ratio: `1.2918648874782441` (bound `1.05`)
- request wall speedup: `0.9183776011887692` (bound `1.05`)

The parked decode trace forces every prefill allocation onto TT-Metal's
allocation-under-active-trace path, which costs about `1.27` seconds per B=4
prefill. That exceeds the `0.85`-second trace capture it would replace, so the
net request wall time regresses by about eight percent.

Both routes also exceeded the `1.25` duplicate-root stability bound
(`1.29943584821287` recapture, `1.293629083003451` reuse), which independently
blocks admission.

## Decision

No production change is selected. The fail-closed transition policy, profile,
wrapper, and evaluator stay in the repository with reuse disabled. The one
prior production admission (B=4 batching with per-group recapture) remains the
admitted behavior, and the deployed production unit was never replaced during
this control.

## Run artifacts

- Raw roots, server logs, and the exact rejection receipt: `.pi/evidence/qwen38/trace-reuse-20260826/`
- One earlier partial attempt (first physical execution) was invalidated by a
  harness timing artifact: a width-3 group assembled when one concurrent client
  missed the coordinator's 10 ms collection window, and explicit mode correctly
  rejected it before any device work. The runner then gained bounded group
  retries, and the complete control reran from scratch.

## Non-claims

- This rejection does not falsify same-width reuse on runtimes other than TT-Metal `0.74.0`.
- It does not measure B=1 single-stream reuse, where the trace-prepare share of wall time is smaller.
- It does not admit any other scheduling behavior, precision, or mesh.
- The parked-trace prefill cost is attributed by phase timing only, not by device-side profiling.
