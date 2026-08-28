## Context

The fast worker writes one JSON receipt after each successful execution. Its `persistent_decode` section contains detailed identities and lifecycle evidence. Operators need a stable aggregate event that supports canary dashboards without coupling them to the full evidence schema.

The exact goal is a versioned, receipt-native telemetry event for every successful fast-worker run. Completion requires pure summary logic, strict cross-field validation, positive and negative tests, and one retained runtime receipt.

False completion includes renaming existing fields, emitting unversioned data, copying high-cardinality payloads, or reporting timings without defined boundaries. This change does not add a network exporter, an alert service, or an automatic admission-disable controller.

The bounded review uses three serial lenses because subagent consent is absent:

1. Reuse the detailed receipt directly.
2. Send metrics to an external exporter from the runtime.
3. Emit a stable aggregate event inside the existing receipt.

## Decisions

### Decision: Emit one aggregate event inside each worker receipt

**Choice:** Add a top-level `persistent_decode_telemetry` object with schema version `1` and a fixed event kind.

**Rationale:** Receipt-native telemetry adds no network dependency and preserves one atomic output. A dedicated object gives monitoring tools a stable contract without exposing the detailed evidence section.

The direct-reuse option keeps no stable boundary and includes high-cardinality identities. The external-export option adds availability, retry, and credential failure modes to the inference process.

### Decision: Keep summary logic pure

**Choice:** Build and validate a typed telemetry snapshot in `persistent_decode.cpp`. Keep clock reads and JSON output in `main.cpp`.

**Rationale:** Unit tests can exercise all categorical logic and invalid combinations without hardware, files, clocks, or mocks.

### Decision: Define timing boundaries explicitly

**Choice:** Record three finite, non-negative monotonic durations.

- `trace_setup_seconds` covers Metal Trace preparation and capture.
- `trace_replay_seconds` covers the bounded Metal Trace replay.
- `cleanup_seconds` covers acknowledgement through state zeroing and slot release.

Normal dispatch records zero for all three durations. A terminal event before replay records zero replay time and a non-negative cleanup duration.

### Decision: Use bounded categorical values

**Choice:** Record only stable enums and aggregate counts for mode, fallback, parity, cleanup, and terminal outcomes.

**Rationale:** Bounded values prevent telemetry cardinality growth. The event excludes prompts, token IDs, request IDs, paths, and binding identities.

### Decision: Cross-check telemetry against detailed evidence

**Choice:** The Rust guard validates the telemetry schema and compares it with the detailed `persistent_decode` section.

**Rationale:** A telemetry event cannot claim persistent execution, parity, or cleanup when the detailed receipt disagrees.

## Risks / Trade-offs

- Receipt-native telemetry requires a collector to read worker receipts. This change does not provide transport.
- A fatal runtime error can stop receipt creation. Process supervision must report that error until a separate failure-receipt design exists.
- Timing fields add monotonic measurements outside the trace timing path. They do not change admission or performance thresholds.
