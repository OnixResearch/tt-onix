## ADDED Requirements

### Requirement: Host timeline boundaries avoid Python call profiling

r[tenstorrent.qwen38_p150x2.host_timeline_signposts] Host timeline diagnostics MUST use bounded Tracy signpost messages without enabling Python call profiling. The comparator MUST validate one ordered start/end pair and MUST retain duplicate stability gates. Results MUST remain diagnostic-only.

#### Scenario: Signpost capture is stable

- GIVEN matched balanced physical roots with valid signposts and C++ enqueue zones
- WHEN duplicate validation passes
- THEN the comparator MAY report host attribution
- AND it MUST NOT admit an optimization.

#### Scenario: Signposts or duplicates are invalid

- GIVEN missing, misordered, or repeated signposts, or unstable duplicate roots
- WHEN validation runs
- THEN the comparator MUST reject attribution
- AND production MUST remain eager.
