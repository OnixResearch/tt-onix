## Context

The rejected physical receipt reported a candidate TTFT ratio of `1.2164174383938555` against a maximum of `1.05`.
The same receipt reported exact token parity, clean resets, and a strong decode-rate gain.

Both eager and candidate paths start TTFT timing immediately before `prefill_traced_chunked`.
Both stop timing after device synchronization, full first-logit readback, and host argmax.
Trace capture starts after that interval.
Request reset also runs before that interval.

The material static difference is page-table and cache capacity.
Eager prefill sizes its page table for the request.
The candidate preallocates maximum sequence capacity and passes that maximum-width table into prefill.

## Goal and Success Contract

The goal is a validated traced-decode candidate whose TTFT ratio is at most `1.05`.

Completion requires:

- exact generated token and text parity;
- clean reset and stable trace bindings;
- request-bounded prefill page-table width;
- unchanged maximum-width trace-bound decode table identity;
- finite phase timing in every measured response;
- order-balanced matched physical evidence on device IDs `0` and `1`;
- all existing decode, TTFT, tail, cleanup, and identity gates;
- production admission only after the new receipt passes.

A precise rejected receipt is also a valid terminal result.

## Architecture

### Functional core

The pure core receives prompt tokens, generation tokens, KV block size, and trace capacity.
It returns the required prefill block count or a domain error.

The core rejects zero, negative, overflow, and capacity-exceeding inputs.
It performs no I/O and reads no ambient state.

### Imperative shell

The server measures phase boundaries and asks the adapter for a bounded prefill table.
The adapter slices its host table without replacing or mutating the trace-bound device table.

The shell records observed timings and widths in response evidence.
The receipt collector verifies and aggregates those observations.

## Decisions

### Decision: Separate prefill width from decode trace capacity

**Choice:** Use an exact request prefix for prefill and retain maximum capacity for decode replay.

**Rationale:** Prefill needs only blocks that the bounded request can address.
Decode trace inputs need one stable shape and address across requests.
Both tables map logical block `n` to physical block `n`.

### Decision: Keep TTFT boundaries unchanged

**Choice:** Preserve the existing start and stop points.

**Rationale:** Moving work outside the interval would hide latency instead of reducing it.

### Decision: Add phase evidence before admission

**Choice:** Record reset, prefill device, readback, trace preparation, and total TTFT separately.

**Rationale:** High variance can produce a false causal claim from aggregate TTFT alone.

### Decision: Use order-balanced physical runs

**Choice:** Run matched `eager, candidate, candidate, eager` service roots.

**Rationale:** Service load order, compilation state, and device history can affect the short-prompt measurements.

## Validation Strategy

Positive core tests cover exact, rounded-up, and maximum-capacity block counts.
Negative tests cover invalid sizes, empty requests, and capacity overflow.

Shell and adapter checks cover a narrow prefill prefix with an unchanged decode table and binding identity.
Receipt self-tests cover complete phase evidence and reject missing, non-finite, or inconsistent widths.

A physical run uses the existing four-prompt corpus, one warm-up, three measured runs, and `64` generated tokens.

## Risks and Mitigations

- **Wrong cache mapping:** Use a prefix of the same monotonic logical table.
- **Trace binding drift:** Keep the full decode device table unchanged and validate its address.
- **Hidden latency:** Keep total TTFT boundaries unchanged.
- **Measurement noise:** Use phase timing and balanced execution order.
- **False admission:** Keep production eager until every receipt gate passes.

## Non-Goals

- Claiming that page-table width is the cause before physical discrimination.
- Reducing trace capture time.
- Changing prefill kernels.
- Generalizing beyond the named P150x2 profile.
