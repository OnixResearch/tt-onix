## ADDED Requirements

### Requirement: Host timeline roots use bounded repeated sampling

r[tenstorrent.qwen38_p150x2.host_timeline_sampling] Each host-timeline root MUST contain at least five complete ordered prefill signpost intervals after warm-up. The comparator MUST calculate each interval independently, use the median interval metrics as the root summary, reject malformed signpost pairs, and retain the `1.25` duplicate-root bound. Physical execution MUST retain two roots per mode in exact `eager,candidate,candidate,eager` order. Results MUST remain diagnostic-only.

#### Scenario: Repeated roots are stable

- GIVEN five valid samples in each immutable physical root
- WHEN both duplicate comparisons pass
- THEN the receipt MAY report diagnostic host attribution
- AND it MUST NOT admit an optimization.

#### Scenario: Sampling evidence is invalid

- GIVEN too few samples, malformed pairs, identity drift, or an unstable duplicate metric
- WHEN validation runs
- THEN the receipt MUST reject attribution
- AND production MUST remain eager.
