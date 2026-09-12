## Context

The service declares `MAXIMUM_ACTIVE_BATCH_WIDTH = 16`, allocates its persistent
decode state at that width, and has never driven it. The cap turned out to be the
closed admission width set `(4, 8)` in both the service policy and the receipt
builder: a B=16 profile is rejected with `workload widths are unsupported`, and a
receipt for one is rejected with `profile widths are unsupported`. No runtime
limit was involved.

## Decisions

### Decision: Widen the closed set rather than special-case a width

**Choice:** Add width 16 to the allowed candidate widths in the service policy
and the receipt builder, and add a B=16 self-test case to the builder.

**Rationale:** The adapter already declares 16 as its capacity and the model
already allocates for it. Keeping the policy set narrower than the capacity is
what made the declared width unreachable, and a special case would leave the two
inconsistent again.

### Decision: Reuse the shared bounds, do not invent new ones

**Choice:** The B=16 profile carries the same four bounds as B=4 and B=8.

**Rationale:** A new width is not a reason to invent a new efficiency floor.
Whatever B=16 scores, it is scored against the same contract, so the result is
comparable to the other widths instead of flattering this one.

### Decision: A sixteen-prompt corpus, in policy order

**Choice:** Extend the eight-prompt corpus to sixteen and generate one request
per prompt.

**Rationale:** A width-sixteen group needs sixteen concurrent requests, one per
corpus prompt. Reusing eight prompts would either double-send requests or
measure a narrower group than the profile claims.

### Decision: Keep serving unchanged and stay unadmitted until the gates pass

**Choice:** Add the width and its evidence path; do not enable batched serving in
production and do not move any binding on a failing receipt.

**Rationale:** The measured result fails per-user retention and TTFT. Those are
served-quality bounds; admitting the width on aggregate throughput alone would
trade per-user quality for a headline number.

## Risks / Trade-offs

- Width 16 aggregates well and is inefficient per user. Both are recorded.
- An intermittent program error fails a whole sixteen-user group and is recovered
  only by retry; the service does not retry, so width 16 stays unadmitted until it
  is root-caused.
- The TTFT failure is expected to improve only if prefill stops being serialized,
  which is the grouped-prefill change's problem, not this one's.
