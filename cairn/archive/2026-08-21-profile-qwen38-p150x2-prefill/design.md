## Context

The rejected traced-decode receipt measured a TTFT ratio of `1.1069467490614955` against the maximum `1.05`. Candidate device prefill was `1.171225828002207` seconds. Eager device prefill was `1.0575300640048226` seconds. First-token readback differed by less than `0.002` seconds.

The package-matched TT-Metal runtime exposes `ProgramRealtimeRecord` callbacks. Each record contains a runtime ID, chip ID, raw timestamps, clock frequency, and kernel-source paths. The callback runs on a receiver thread and can lag device synchronization.

## Goals

- Attribute eager and candidate prefill time to exact program signatures.
- Preserve vendor types at the adapter boundary.
- Produce deterministic summaries from explicit values.
- Keep profiling overhead and evidence outside admission timing.
- Fail closed when profiling is unavailable or malformed.

## Non-goals

- Do not admit traced decode.
- Do not change eager production behavior.
- Do not claim that summed program durations equal wall-clock TTFT.
- Do not select a kernel configuration before physical attribution exists.
- Do not use simulator or static source data as physical performance evidence.

## Decisions

### Decision: Use the package-matched real-time program profiler

**Choice:** Register the TTNN real-time profiler callback only when the operator enables the diagnostic flag.

**Rationale:** The callback gives device timestamps and kernel-source identities without adding a synchronization after every operation. Host timers cannot attribute asynchronous device work. Per-operation synchronization would change the execution schedule being measured.

### Decision: Keep a pure summary core and an imperative adapter

**Choice:** A pure core validates domain records and groups durations by exact kernel-source signature and physical chip. A narrow adapter owns callback registration, locking, receiver drain, and vendor translation. The service shell starts and ends one prefill region.

**Rationale:** This keeps policy and arithmetic deterministic. TTNN types, threads, sleeps, and callback lifecycle remain outside the core.

### Decision: Treat profile responses as diagnostic only

**Choice:** A profiled response records `measurement_mode = diagnostic-prefill-profile` and `admission_eligible = false`. The TTFT receipt path rejects that mode.

**Rationale:** The package-matched profiler tests require a receiver drain after device synchronization. That drain can add latency before readback. Profiled TTFT is not comparable with normal admission TTFT.

### Decision: Compare exact program signatures by chip

**Choice:** A signature is the sorted, unique tuple of non-empty kernel-source paths. The summary reports record count, total device duration, maximum record duration, and each chip total. It sorts signatures by decreasing maximum chip total and then lexically.

**Rationale:** Runtime IDs identify cached programs and can repeat. Per-chip totals avoid unsupported assumptions about callback interleaving or cross-chip occurrence pairing.

### Decision: Use one bounded physical profiling round

**Choice:** Run roots in the exact order `eager,candidate,candidate,eager`. Use the existing four prompts, one warm-up, three measured requests, and `64` generated tokens. Compare only like-for-like profiled roots.

**Rationale:** This preserves the existing corpus and limits run-order bias. A later unprofiled admission round remains separate.

## Error handling

- Requested profiling fails startup when the callback API or active profiler is unavailable.
- Nested or unfinished profile regions fail before dispatch.
- Empty records, invalid chips, invalid timestamps, invalid frequencies, or empty kernel sources fail summary validation.
- Callback unregister failure is an adapter error and prevents a successful diagnostic claim.
- The comparator rejects missing roots, wrong execution order, identity drift, admission-eligible profile data, or unmatched prompt counts.

## Risks / Trade-offs

- The profiler can perturb device resources. Therefore, only eager-versus-candidate differences from the same profile mode are comparable.
- A fixed receiver drain can still miss delayed records. The summary reports record counts, and duplicate roots must agree on signature coverage before attribution.
- Kernel-source signatures identify programs, not model-layer names. A source lookup is required before an optimization is selected.
- Program-duration sums are diagnostic attribution. They are not a replacement for end-to-end TTFT.
