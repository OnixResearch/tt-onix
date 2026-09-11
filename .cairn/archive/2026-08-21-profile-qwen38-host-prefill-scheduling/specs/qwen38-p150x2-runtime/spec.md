## ADDED Requirements

### Requirement: Host timeline diagnostics use package-matched tooling and stay diagnostic

r[tenstorrent.qwen38_p150x2.host_timeline_profile] The repository MAY provide host-side Tracy CLI tooling built from the package-matched TT-Metal vendored fork so the packaged Python `tracy` module can capture host zones. Host timeline capture MUST be an operator-enabled diagnostic that runs only under authorized device isolation. Host timeline observations MUST NOT satisfy any admission gate. A host attribution claim MUST come from a deterministic comparator over validated captures from matched balanced physical roots.

#### Scenario: Matched captures expose host spans

r[tenstorrent.qwen38_p150x2.host_timeline_profile.valid]

- GIVEN validated captures from matched `eager,candidate,candidate,eager` physical roots with consistent runtime, package, model, and mesh identities
- WHEN the deterministic comparator evaluates exported host zones
- THEN it MUST report ranked inter-enqueue gaps and phase totals per decode mode
- AND it MAY name one host-side attribution result without admitting any optimization.

#### Scenario: Host timeline evidence is invalid

r[tenstorrent.qwen38_p150x2.host_timeline_profile.invalid]

- GIVEN a missing capture, malformed zone record, identity drift, or wrong execution order
- WHEN comparator validation runs
- THEN it MUST reject the attribution result
- AND production MUST remain unchanged.

#### Scenario: Host profiling is not enabled

r[tenstorrent.qwen38_p150x2.host_timeline_profile.disabled]

- GIVEN the production eager service or an unprofiled candidate service
- WHEN requests execute
- THEN the service MUST NOT run a capture tool or change its response payloads
- AND existing TTFT boundaries and admission behavior MUST remain unchanged.
