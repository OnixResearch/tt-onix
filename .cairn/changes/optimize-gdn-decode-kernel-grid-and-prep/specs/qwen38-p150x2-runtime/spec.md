## ADDED Requirements

### Requirement: GDN decode layer occupancy and host preparation are optimized behind exact parity

r[tenstorrent.qwen38_p150x2.gdn_decode_kernel_efficiency]

The repository MAY replace the admitted GDN decode kernel with a variant that raises per-layer grid occupancy and removes host-side preparation ops. The variant MUST split each value head's recurrent state across cores, and it MAY pipeline head groups where the state split alone leaves occupancy below the profile-owned floor. The variant MUST fold the fp32 typecasts, the L2 norms, and the query scale into the reader or the compute kernel, and it MUST NOT leave them as separate device operations. The variant MUST preserve the fp32 input and state type. A candidate MAY be admitted only after a matched physical screen proves exact decode-output and recurrent-state parity against the admitted reference, an occupancy at or above the profile floor, and a step-time ratio at or below the profile bound. The screen MUST reject on unstable duplicate roots. The screen MUST NOT select any candidate whose parity fails, and MUST NOT admit the T=K+1 verify path, a reduced precision type, a new batch width, or a new serving route.

#### Scenario: The optimized kernel passes exact parity and both bounds

r[tenstorrent.qwen38_p150x2.gdn_decode_kernel_efficiency.valid]

- **GIVEN** the occupancy-split and prep-folded kernel, a warmed model, and matched control roots
- **WHEN** the screen runs in exact `control,candidate,candidate,control` order
- **THEN** every decode token and the recurrent state MUST match the admitted reference exactly
- **AND** the derived occupancy MUST be at or above the profile floor
- **AND** the candidate step-time ratio MUST be at or below the profile bound
- **AND** the receipt MUST record how occupancy was derived.

#### Scenario: Parity or occupancy fails and the admitted kernel stays

r[tenstorrent.qwen38_p150x2.gdn_decode_kernel_efficiency.invalid]

- **GIVEN** a state or output mismatch, a duplicate-root spread above the stability bound, an occupancy below the floor, or a step-time ratio above the bound
- **WHEN** the deterministic evaluator reads the screen
- **THEN** it MUST reject the candidate with the failing gate named
- **AND** the admitted kernel MUST remain the served route.

#### Scenario: A malformed occupancy or head-group plan is rejected

r[tenstorrent.qwen38_p150x2.gdn_decode_kernel_efficiency.plan]

- **GIVEN** a zero core count, a state split that does not partition the state, overlapping head groups, a head group above the closed size range, or a plan outside the closed set
- **WHEN** the pure occupancy core validates the plan
- **THEN** it MUST fail before any device work
- **AND** the receipt MUST name the bounded rejection reason.
