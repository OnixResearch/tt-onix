# Physical host-timeline result

Date: 2026-08-21

## Scope

- Physical mesh: exact `1x2`, PCIe/UMD IDs `0` and `1`.
- Runtime: `/nix/store/mg0ha5nsvnir1y1ysiyyzmm4hk83z3m6-tt-metal-0.74.0`.
- Package: `/nix/store/k7pnnn0h5lc5206aw7ylsnqpa5yzvyii-qwen36-0.1.0`.
- Model/checkpoint: `Qwen3.8-27B` at
  `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
- Execution order: exactly `eager,candidate,candidate,eager`.
- Each root used one compile warm-up outside capture and one measured request
  inside capture. Each response generated one token.
- Every measured response reported
  `measurement_mode = diagnostic-host-timeline` and
  `admission_eligible = false`.

An initial root that captured compile warm-up stalled under unbounded trace
traffic. It was stopped, excluded, and production was restored before the valid
bounded run. The final runner starts capture only after compile warm-up.

## Result

The package-matched Tracy tools worked. Each valid root contained the bounded
Python `_profiled_prefill_device` zone plus C++ `EnqueueProgram` and
`FDMeshCommandQueue::finish` zones.

Candidate duplicates were stable:

- prefill-wall ratio: `1.0266615508948504`;
- enqueue-execution ratio: `1.0760504091447658`;
- inter-enqueue-gap ratio: `1.0197062819641554`.

Eager duplicates were not comparable:

- prefill-wall ratio: `2.900386615837245`;
- enqueue-execution ratio: `9.206084094193452`;
- inter-enqueue-gap ratio: `1.8710775379330478`.

The deterministic comparator permits a maximum duplicate ratio of `1.25` and
therefore returned:

- verdict: `rejected`;
- reason: `duplicate-root-instability`.

Receipt:
`pkgs/qwen36/evidence/2026-08-21/qwen38-27b-host-timeline-attribution-receipt.json`

Receipt BLAKE3:
`e9fc1b4aeec6deb9caa2a9390b28c3f00f9e4e3d1fb9870ca1199001f43e734c`

## Decision

No optimization is selected. The Python partial-profiler boundary perturbs or
queues enough eager host work to invalidate duplicate attribution. The
candidate-minus-eager values in the rejected receipt are diagnostic artifacts,
not causal evidence.

The smallest next check is a lower-overhead boundary that emits C++ Tracy
messages or signposts without enabling Python call profiling. It must retain
the existing C++ enqueue/finish zones and pass duplicate stability before
another attribution claim.

Production was restored with `ttwkv7-owner-control restore`. The root-owned
`qwen38-p150x2.service` is active on port `8000`; production remains eager.
