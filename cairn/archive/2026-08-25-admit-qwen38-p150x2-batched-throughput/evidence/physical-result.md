# Physical result

## Verdict

The bounded B=4 throughput profile passes every predeclared batch-admission gate on physical P150x2 devices 0 and 1.

This result admits only the exact B=4 active-prefix profile. It does not admit B=8, B=16, arbitrary slot recycling, live prefill during decode, or the separate single-stream eager-versus-traced profile.

## Execution

- Order: `single,batch,batch,single`
- Roots per route: 2
- Prompts per root: 4
- Warm-up runs per prompt: 1
- Measured runs per prompt: 3
- Generated tokens: 64
- Measured responses: 48
- Package: `/nix/store/ibjk9b3lynppx1swbvi6x9qg07pdkpp8-qwen36-0.1.0`
- Runtime: `/nix/store/mg0ha5nsvnir1y1ysiyyzmm4hk83z3m6-tt-metal-0.74.0`
- Checkpoint and tokenizer: `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`

The valid run followed one excluded startup attempt. That attempt lacked `HOME` in the transient systemd environment and crashed during TTNN import. It produced no response root. The runner added the same root home environment used by the deployed service, then restarted the complete balanced order.

## Correctness

- Exact generated token parity: pass
- Exact generated text parity: pass
- Clean reset: pass
- Complete active prefix: pass
- Distinct state slices: pass
- Physical device identity: pass
- Package, runtime, model, profile, and workload identity: pass
- Production restoration and readiness: pass

## Performance

| Metric | Single | B=4 batch | Ratio | Bound |
| --- | ---: | ---: | ---: | ---: |
| Aggregate decode | 16.1424 t/s | 57.3318 t/s | 3.5516 | at least 3.0 |
| Per-user decode | 16.1424 t/s | 14.3330 t/s | 0.8879 | at least 0.75 |
| Internal TTFT | 1.3619 s | 4.6800 s | 3.4363 | at most 4.5 |
| Tail token latency | 0.07545 s | 0.09208 s | 1.2203 | at most 1.5 |

The corrected internal TTFT ends when all active first-token winners are available. It excludes later state-slot installation and trace preparation. Those phases remain explicit:

- Single median state installation: 0.20884 seconds
- B=4 median state installation: 0.75937 seconds
- Single median trace preparation: 1.11793 seconds
- B=4 median trace preparation: 0.99647 seconds

## Receipt

- Path: `pkgs/qwen36/evidence/2026-08-25/qwen38-27b-batch-admission-receipt.json`
- BLAKE3: `176d912457f4c0048be90c9b477709e4360dae0681212cf275d108f3ca48465f`
- Admission reasons: none
- `admitted`: `true`

Raw roots remain under `.pi/evidence/qwen38/batch-admission-20260825/`.

## Claim boundary

The receipt admits bounded B=4 aggregate-throughput scaling for the tested execution artifact and workload. The separate single-stream profile remains `physically_admitted = false` because its existing TTFT gate did not pass. The result does not claim streaming, mixed generation lengths, B=8, B=16, arbitrary continuous batching, or another package and runtime identity.
