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

## Follow-up: the batched-prefill alternative was measured and is worse

Because the grouped-prefill route's guards allow it only at exact fill, width 16
is the one configuration where it can run. With `QWEN35_GROUPED_PREFILL=1` the
sixteen-user group returns 16/16 responses with no program error, and its batch
TTFT is **33.095 s against the 23.589 s per-user baseline**. Aggregate decode
throughput is unchanged (166.9 against 168.9 tokens/second), and
`prefill_first_token_seconds` equals `ttft_seconds`, so every user waits for the
whole group.

So the TTFT-ratio failure is not explained by "prefill was never batched": the
batched alternative is slower here at 384 scan instances over 110 cores, and it
removes per-user pipelining by construction. The failure therefore stands on its
own merits, with the mechanism measured rather than assumed.

## V4 follow-up: the intermittent defect narrowed to a 0.77 validator interaction

### What the validator actually checks

`tt_metal/impl/program/program.cpp` (pinned 0.77 source, `validate_circular_buffer_core_ranges`):

```cpp
const bool in_worker_grid = cr.end_coord.x < grid_size.x && cr.end_coord.y < grid_size.y;
TT_FATAL(in_worker_grid || entirely_on_service_cores(cr), ...);
```

`CoreRange::str()` renders `[start.x-start.y - end.x-end.y]`
(`tt_metal/common/core_coord.cpp:115`). So the recorded ranges are **vertical**:
`(0,0)-(0,10)` is eleven rows in column 0, and the physical worker grid is
`11 x 10`, i.e. `y` may only reach 9. A CB range is exempt when *every* core in
it is claimed through `ServiceCoreManager` — the exemption that collective ops
rely on.

### Why the width is not the trigger

A clean single-server probe fired groups of 10, 11, 13 and 16 concurrent
requests. The coordinator assembled groups of 3, 10, 10, 11 and 15 from those
bursts and **all of them failed** with `BatchDecodeAdapterError`, while a
width-1 request **passed**. Width 1 passed because its decode trace was already
resident from warm-up (widths 1 and 4 are warmed); every other width needs a
**new trace capture**, and the fatal happens inside that capture.

So the trigger is trace capture at a width whose trace is not yet resident — not
the width itself, and not a count proportional to the group size. That also
explains the intermittency seen in the matched run: a capture attempt can fail
and a later attempt at the same width can succeed, which is why the batch roots
completed with whole-group retries.

### What this points at

- The model's own shard grids in the GDN attention path are **horizontal**
  (`CoreRange(CoreCoord(0,0), CoreCoord(N-1, 0))`, two sites in
  `ttnn_gated_attention.py`), so they are not the source of a vertical range.
- The range is exactly `grid.x` (11) rows tall, which is the signature of a
  transposed grid argument, or of a CB laid out over a *service-core column*.
- The validator's exemption is claim-set based and, per its own comment, was
  recently extended to flatten `MeshDevice` and query `ServiceCoreManager` per
  chip. A layout that 0.74 accepted can therefore fail on 0.77 when the claim set
  and the CB range disagree.

The remaining step is C++-side: identify the op that creates program 1324 (or
2360/2396 in the matched run). TTNN's op logging is not usable for this — it
inserts event synchronization, which trace capture forbids
(`TT_FATAL: Event Synchronization is not supported during trace capture`) — so the
instrument has to be a device/program-level trace rather than python logging.

### Practical consequence

- Width 16 executes and is correct but is not reliable at new trace widths; the
  binding stays unadmitted and the service coordinator does not retry, so serving
  must not be enabled on it.
- Any measurement of a new width should capture its trace before the measured
  groups (the harness's group retry already absorbs the first-capture failure).

## V4 addendum: the measurement is genuine, and the failure does not scale with width

Group widths actually observed in the matched roots:

| Root | width-16 groups | sub-groups seen | program fatals |
| --- | ---: | --- | ---: |
| batch-1 | 64 | 10, 11 x3, 5 x3, 6, 7, 9 | 3 (all `(0,0)-(0,10)`) |
| batch-2 | 64 | none | 0 |

So the width-16 receipt is built from **64 genuine width-16 groups in each batch
root** (64 = 16 warmup + 48 measured), and the earlier concern that the coordinator
was silently serving smaller groups is excluded. The single roots never fataled.

Two facts sharpen the defect and correct a first reading of it:

1. The offending range was `(0,0)-(0,10)` in **every** fatal of the matched run —
   eleven rows, which is exactly `grid.x` — while the failed group widths were
   11, and in the separate probe 3, 10, 11 and 15. The range therefore does not
   scale with the group width; it is a fixed eleven-core column (thirteen in one
   probe instance).
2. Failures and 64 successes interleave inside one process, so a fatal does not
   always poison the session — but in the width probe it did (49 subsequent
   request failures, one pass). Whether a session recovers appears to depend on
   how the failing program is subsequently re-created.

Combined with the trigger being a non-resident trace width, the remaining
candidate mechanisms are a transposed grid coordinate for one program family, or
a CB laid out along a service-core column whose claim set does not cover it.
Identifying which needs a program-level trace; python-side logging is unusable
because trace capture rejects event synchronization.
