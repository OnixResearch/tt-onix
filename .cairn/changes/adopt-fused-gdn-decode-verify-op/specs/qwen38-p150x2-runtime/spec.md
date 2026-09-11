## ADDED Requirements

### Requirement: Fused GDN decode and verify is gated, screened, and default-off

r[tenstorrent.qwen38_p150x2.fused_gdn_decode_op]

The repository MAY route the GDN T=1 decode step and the T=K+1 verification step through a fused device op instead of the composite graph. The gate MUST default off. An absent, empty, non-string, or unknown gate value MUST select the composite path. With the gate off, the produced decode graph MUST be byte-identical to the current admitted graph. The repository MUST record the exact upstream source for the vendored op, including the pull request or branch, the commit, and the review state, and it MUST NOT treat an upstream throughput claim as local evidence. A candidate MAY be admitted only after a matched physical screen passes. The screen MUST verify exact greedy token parity and recurrent-state parity against the composite reference before it reads any timing, and it MUST then require a step-time ratio at or below the profile-owned bound. The screen MUST reject on unstable duplicate roots. A passing screen MUST NOT admit reduced precision, a new batch width, a new serving route, or the MTP speculative path.

#### Scenario: The fused path passes the physical screen

r[tenstorrent.qwen38_p150x2.fused_gdn_decode_op.valid]

- **GIVEN** the fused op vendored against the pinned runtime, a warmed model, and matched control roots
- **WHEN** the screen runs in exact `control,candidate,candidate,control` order
- **THEN** every greedy token and the recurrent state MUST match the composite reference exactly
- **AND** the candidate step-time ratio MUST be at or below the profile bound
- **AND** the receipt MUST bind the selected kernel, the gate value, and the execution identity.

#### Scenario: Unsupported shape or missing op selects the composite path

r[tenstorrent.qwen38_p150x2.fused_gdn_decode_op.fallback]

- **GIVEN** an absent op binding, an unsupported shape, a dtype outside the closed set, or a malformed gate value
- **WHEN** the pure selection core resolves the decode mode
- **THEN** it MUST select the composite path before any device work
- **AND** the receipt MUST name the bounded fallback reason.

#### Scenario: Screen evidence is invalid or incomplete

r[tenstorrent.qwen38_p150x2.fused_gdn_decode_op.invalid]

- **GIVEN** a token mismatch, a state mismatch, a missing root, a duplicate-root spread above the stability bound, or a step-time ratio above the bound
- **WHEN** the deterministic evaluator reads the screen
- **THEN** it MUST reject the candidate with bounded reasons
- **AND** the gate MUST stay off and the composite path MUST remain the served route.

#### Scenario: MTP economics are re-evaluated under the fused verify kernel

r[tenstorrent.qwen38_p150x2.fused_gdn_decode_op.economics]

- **GIVEN** a passed decode screen and the fused T=K+1 verify kernel
- **WHEN** the existing MTP drafter and rollback probes run under the same predeclared bounds
- **THEN** the receipt MUST report emitted-token economics and acceptance-outcome parity separately
- **AND** a changed acceptance outcome MUST reject the candidate even when the aggregate speedup is positive.
