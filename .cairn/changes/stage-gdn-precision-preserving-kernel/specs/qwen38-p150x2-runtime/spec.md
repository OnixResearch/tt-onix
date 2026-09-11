## ADDED Requirements

### Requirement: A staged GDN recurrence reproduces the composite precision boundaries

r[tenstorrent.qwen38_p150x2.gdn_precision_staging]

The repository MAY replace the fused GDN decode op's single-stage fp32 recurrence with a staged kernel architecture that reproduces each precision boundary the composite recurrence performs. The stage order and the boundary list MUST be owned by a pure, device-free core that rejects an unknown stage, a missing boundary, a duplicated boundary, and an out-of-order stage list. The staged path MUST default off, and an absent, empty, non-string, or unknown gate value MUST select the existing fused route. Greedy and deterministic serving MUST keep exact token parity as their admission gate; the existing sampled-only distributional contract MUST NOT be extended to them by this change. A candidate MAY be admitted only after a matched physical screen proves exact greedy token parity over a fixed multi-prompt corpus and passes the existing GDN ratio and full-step ratio bounds. The screen MUST reject on unstable duplicate roots. A passing screen MUST NOT admit a reduced precision type, a new batch width, or a new serving route.

#### Scenario: The staged recurrence passes exact greedy parity and both speed bounds

r[tenstorrent.qwen38_p150x2.gdn_precision_staging.valid]

- **GIVEN** the staged kernel, a warmed model, a fixed multi-prompt corpus, and matched control roots
- **WHEN** the screen runs in exact `control,candidate,candidate,control` order
- **THEN** every greedy token and the full generated text MUST match the composite reference exactly
- **AND** the GDN ratio MUST be at or below its bound
- **AND** the full-step ratio MUST be at or below its bound
- **AND** the receipt MUST name the stage order and the boundary list it exercised.

#### Scenario: The staged design cannot reproduce parity within budget

r[tenstorrent.qwen38_p150x2.gdn_precision_staging.exhausted]

- **GIVEN** a bounded attempt whose staged design cannot reproduce exact greedy parity
- **WHEN** the attempt reaches its declared budget
- **THEN** the change MUST record an exhausted result with the discriminating evidence
- **AND** the fused route MUST stay restricted to sampled serving
- **AND** no parity bound MAY be relaxed to convert the result.

#### Scenario: A malformed staging plan is rejected

r[tenstorrent.qwen38_p150x2.gdn_precision_staging.invalid]

- **GIVEN** an unknown stage, a missing boundary, a duplicated boundary, an out-of-order stage list, or a gate value outside the closed set
- **WHEN** the pure staging core validates the plan
- **THEN** it MUST fail before any device work
- **AND** the existing fused route MUST remain the selected path.
