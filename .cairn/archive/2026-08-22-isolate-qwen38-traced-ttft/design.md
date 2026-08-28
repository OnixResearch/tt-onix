## Context

The candidate calls `_prefill_first_token` before `ensure_trace`. Its TTFT clock therefore excludes decode trace capture. The eager path uses request-sized paged KV caches. The candidate keeps trace-capacity caches across requests. Measured candidate requests also run while a decode trace is parked.

Prior program profiling found only a `2.306050567859769` millisecond device-program delta inside a much larger wall-prefill gap. Prior host attribution failed duplicate-root stability. The next control must remove model-load and process-root differences while preserving the exact package and physical devices.

## Decisions

### Decision: use one model and two stable KV cache sets

**Choice:** Load the pinned model once. Allocate one request-sized cache set and one trace-capacity cache set. Bind one set before each route without changing the bounded page-table width.

**Rationale:** The model, weights, compilation cache, prompt, GDN buffers, process, and device ownership remain fixed. Only the paged-KV capacity changes for the first comparison.

### Decision: use one overlapping balanced route schedule

**Choice:** Run roots in this order:

1. request-sized cache without a parked trace;
2. trace-capacity cache without a parked trace;
3. trace-capacity cache with a parked trace;
4. trace-capacity cache with a parked trace;
5. trace-capacity cache after trace release; and
6. request-sized cache without a parked trace.

Each root contains one unmeasured warm-up and at least five retained samples. The cache comparison reads roots `1,2,5,6` as `request,capacity,capacity,request`. The parked-trace comparison reads roots `2,3,4,5` as `untraced,traced,traced,untraced`.

**Rationale:** One schedule gives two balanced comparisons. It also measures the untraced full-capacity route on both sides of trace parking.

### Decision: reset state outside the TTFT clock

**Choice:** Bind the selected cache set, zero GDN state in place, and synchronize before each sample. Then measure device prefill, first-token readback, and their unchanged TTFT sum.

**Rationale:** Existing admission excludes request reset from TTFT. The control keeps the same boundary and prevents queued reset work from entering prefill time.

### Decision: evaluate evidence in a pure core

**Choice:** The core validates identities, route order, cache widths, trace state, token parity, durations, root sample counts, within-root spreads, and duplicate-root ratios. It computes cache-capacity, parked-trace, and combined TTFT ratios from root medians.

**Rationale:** The TTNN adapter owns effects and timing. The pure core owns the causal decision and fail-closed evidence rules.

### Decision: keep the result diagnostic

**Choice:** A ratio more than `1.05` can select a bounded mechanism for a later correction. The control cannot admit traced decode. A later unprofiled matched admission run remains mandatory.

**Rationale:** Same-process evidence removes process drift but does not reproduce the complete production service boundary.

## Risks / Trade-offs

- Switching stable cache bindings does not reproduce fresh process allocation behavior.
- A parked trace can affect resources beyond the measured cache tensors.
- Independent cache sets can change DRAM placement.
- Full-logit first-token readback remains part of both matched routes.
- Trace release or cleanup failure can leave the production service unavailable.
- One model and two outages cannot establish long-context, concurrency, batch, or thermal behavior.
