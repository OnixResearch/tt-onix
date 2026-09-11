## ADDED Requirements

### Requirement: BF4 weight cache families are gated, per-family evidenced, and default-off

r[tenstorrent.qwen38_p150x2.bf4_weight_cache_candidates]

The repository MAY provide opt-in BF4 weight caches for the LM head, the MLP down projection, and the GDN weight families. Each family MUST keep its own distinct `.bfp4` cache name, and no family MAY overwrite or replace a default BFP8 cache. Each family gate MUST default off, and an absent, empty, non-string, or unknown gate value MUST select BFP8. A family MAY move to production precision only after it passes its own numerical screen and its own matched physical screen, and only after owner review. The numerical screen MUST include exact greedy token and text parity on a fixed corpus and an accuracy floor over a fixed corpus that the profile owns; token parity alone MUST NOT be sufficient. The physical screen MUST report a decode step-time ratio against the BFP8 baseline on matched physical roots in exact `control,candidate,candidate,control` order. A combined or bundled run MUST NOT admit more than one family. The tool MUST NOT infer admission from a flag, a cache format, a PCC value, a static byte count, a simulator result, or an incomplete receipt.

#### Scenario: One family passes its numerical and physical screens

r[tenstorrent.qwen38_p150x2.bf4_weight_cache_candidates.valid]

- **GIVEN** a named BF4 family, its own cache name, a fixed parity corpus, a fixed accuracy corpus, and matched physical roots on devices 0 and 1
- **WHEN** the numerical screen and the matched physical screen complete
- **THEN** exact greedy token and text parity MUST pass
- **AND** the accuracy floor MUST pass
- **AND** the receipt MUST report the measured step-time ratio beside the predicted byte saving
- **AND** only that named family MAY move to production precision.

#### Scenario: A family fails parity, the accuracy floor, or the timing bound

r[tenstorrent.qwen38_p150x2.bf4_weight_cache_candidates.invalid]

- **GIVEN** a token mismatch, a text mismatch, an accuracy score below the floor, an unstable duplicate root, or a step-time ratio above the bound
- **WHEN** the deterministic evaluator reads the screens
- **THEN** it MUST reject that family with the failing gate named
- **AND** BFP8 MUST remain the production precision for that family
- **AND** every other family MUST be reported separately.

#### Scenario: A cache name or gate value is unsafe

r[tenstorrent.qwen38_p150x2.bf4_weight_cache_candidates.cache]

- **GIVEN** a BF4 cache name that collides with a BFP8 cache, a missing `.bfp4` suffix, a gate value outside the closed set, or a run where a default BFP8 bin changed
- **WHEN** the pure cache-name and gate policy core validates the plan
- **THEN** it MUST fail before device work or before any measured window
- **AND** the default BFP8 caches MUST stay byte-identical.
