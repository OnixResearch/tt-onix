# Physical result

## Verdict

The device-resident prefill staging change **PASSES** the full B=4 batch
admission on its new execution identity. B=4 batching is re-admitted with the
staged path.

- Receipt BLAKE3: `972a5cc8405b5aec1443ff72e86a2d62b8a68e703d2462991d73ff54b665e382`
- Tracked receipt: `pkgs/qwen36/evidence/2026-08-26/qwen38-27b-batch-admission-receipt.json`
- Measured package: `/nix/store/jca6ipw0xmp4nbjhihg56plzz7ndvvpw-qwen36-0.1.0`
- Promoted package: `/nix/store/yzcwyq8hy3bglyw1720lgl97z8r2xwi1-qwen36-0.1.0`
- Execution identity: `/nix/store/2x7rd7mg91lg07qjsri2ngvlxymrqvkw-qwen36-execution-identity-0.1.0`
- Runtime: `/nix/store/mg0ha5nsvnir1y1ysiyyzmm4hk83z3m6-tt-metal-0.74.0`
- Physical devices: `0` and `1`
- Root order: `single,batch,batch,single`, four prompts, three measured runs per prompt per root

## Measured gates

| Gate | Measured | Bound |
| --- | ---: | ---: |
| Aggregate decode speedup | `3.5210155439006345` | at least `3.0` |
| Per-user decode retention | `0.8802538859751586` | at least `0.75` |
| TTFT ratio | `3.792019652883651` | at most `4.5` |
| Tail-latency ratio | `1.1300310605593398` | at most `1.5` |
| Exact token parity | pass | required |
| State slices 0..3 | pass | required |
| Clean reset | pass | required |

## Improvement over the prior admitted artifact

Phase medians against the previous admitted receipt (`2bf7d74e…`,
package `jilqzkn7…`):

| Phase (median) | Prior | Staged | Delta |
| --- | ---: | ---: | ---: |
| B=1 TTFT | `0.8882480295142159` s | `0.7375790529767983` s | −17.0% |
| B=1 state install | `0.12315196596318856` s | `0.058731721015647054` s | −52.3% |
| B=4 state install | `0.49401886702980846` s | `0.2969129345146939` s | −39.9% |
| B=4 aggregate decode | `58.33` t/s | `57.77` t/s | within noise |

The device-staging patch removed the per-user GDN host round trip (the state
install phase), and the deferred logits readback with one synchronization cut
single-stream TTFT. Aggregate decode is unchanged, as expected: the RoPE table
cache touches host-side input preparation, which is inside the measured decode
interval only marginally.

## Run artifacts

- Raw roots, request corpus, server logs, and the receipt:
  `.pi/evidence/qwen38/prefill-staging-admission-20260826/`
- The remote builder (aspen1) terminated the first package build under memory
  pressure from unrelated concurrent jobs (load 60, 124/125 GiB used); the
  identical derivation then built locally and passed the complete device-free
  check (32 PASS, 0 FAIL).

## Non-claims

- This admission covers the staged B=4 path only; B=8 remains rejected and B=16 remains unmeasured.
- The parked-trace clone hazard is gated, not fixed; paths with a resident bucket or chunk prefill trace keep the host round trip.
- The prior receipt is not transferred; the new receipt binds the new execution identity.
- No deployment claim: production still runs the previously deployed closure until the Onix pin is updated separately.
