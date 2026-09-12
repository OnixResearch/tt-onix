# V3-V5, D1: width 16 executes and is correct, and misses two efficiency bounds

Recorded 2026-09-12 for change `enable-qwen38-p150x2-b16`.

## V2/V3: the matched width-16 run

Four roots in `single,batch,batch,single` order over the sixteen-prompt corpus,
one warmup plus three measured runs per prompt, 64 generated tokens, on physical
devices 0 and 1. All four roots produced 48 measured responses (16 prompts x 3),
and all sixteen concurrent requests returned at `batch_width: 16`.

Receipt: `pkgs/qwen36/evidence/2026-09-12/qwen38-27b-batch-admission-b16-receipt.json`.

### Correctness: every gate passes

```json
{"active_prefix": true, "clean_reset": true, "distinct_state_slices": true,
 "exact_text_parity": true, "exact_token_parity": true}
```

### Performance: two bounds fail

| Bound | Measured | Result |
| --- | ---: | --- |
| aggregate decode speedup >= 6.0 | 10.1393 | pass |
| per-user decode retention >= 0.75 | 0.6337 | **FAIL** |
| TTFT ratio <= 9.0 | 12.6263 | **FAIL** |
| tail-latency ratio <= 1.5 | 1.4900 | pass |

Median timings: batch aggregate 168.89 tokens/second (per-user 10.56) against
single-stream 1.868 seconds TTFT; batch TTFT 23.589 seconds.

**Reading it honestly.** Width 16 is a real aggregate win — 168.9 tokens/second
against roughly 100 at width 8 and roughly 16.7 single-stream — and it is
numerically correct. It is also *inefficient per user*: each user gets 63.4
percent of the single-stream rate, and the batch TTFT ratio is 12.6 because the
service still runs sixteen serialized per-user prefills before the first batched
step. Both failures have the same shape as the B=8 near miss, with width 16
amplifying them. This is the measured reason the binding stays unadmitted.

## V4: the intermittent program error

Observed in the first batch root, three times, and zero times in the second:

```
TT_FATAL: Circular buffer core range [0-0 - 0-12] in program 2396 exceeds device
compute grid (11x10) and is not entirely on cores claimed via SetProgramRun...
```

- The range `(0,0)-(0,12)` spans thirteen rows in column 0; the physical grid is
  eleven columns by ten rows, so the extent exceeds the grid on the y axis.
- Affected program id: 2396.
- Observed rate: three occurrences in batch-1 (each failing a whole sixteen-user
  group, absorbed by the harness's five-attempt group retry), zero in batch-2 and
  in both single roots.
- Consequence: a width-16 group fails as a unit and succeeds on retry. The
  service's coordinator does not retry, so in a serving configuration this would
  surface as a request failure. Until it is root-caused, width 16 is correct but
  not reliable, which is the second reason the binding stays unadmitted.

The program id and the axis differ from the B=32 divisibility defect recorded in
the B=8 change (`per_core_M % out_subblock_h == 0`), so this is a distinct
defect.

## V5: nothing else moved

- The B=4 and B=8 profiles, contracts, bindings, and receipts are untouched
  (`git diff` shows only the new B=16 files, the widened width tuple, and the
  receipt builder's widened width set with its added self-test case).
- Production still serves the single-stream candidate route; no batched serving
  route was enabled, and the service's batch admission stays off.
- The shared and variant packages build, and the device-free check passes.

## D1: the binding records the rejected measurement

`pkgs/qwen36/batch-admission-b16-binding.ncl` keeps `physically_admitted = false`
and records this receipt's digest, so the width-16 state is an exact rejected
measurement rather than an unmeasured one. The install phase enforces the same
consistency rule as B=8: the exported B=16 profile's `physically_admitted` must
equal the tracked receipt's own verdict, and the exported digest must equal the
tracked receipt's `b3sum`.

## Non-claims

- No B=16 admission, and no change to the B=4 or B=8 bindings.
- No serving-route change; production behavior is unchanged.
- The aggregate throughput number is the receipt's own protocol measurement, not
  a serving benchmark, and it is not a claim about long-context or mixed-width
  batching.
- Correctness is exact for this corpus and these prompts; nothing here claims
  arbitrary prompt mixes.

## D2: gates

| Gate | Result |
| --- | --- |
| `repository-quality-cairn` | PASS |
| `qwen38` | PASS |
| `qwen38-nixos-service` | PASS |
| `nix build .#qwen38 .#qwen38-gdn-grid` | exit 0 |
| `./scripts/qwen36-batch-receipt.rs --self-test` | exit 0 over widths 4, 8, and 16 |
| `repository-quality-artifacts` | FAIL, pre-existing manifest drift |

Production `qwen38-p150x2.service` is active and healthy after the run.
