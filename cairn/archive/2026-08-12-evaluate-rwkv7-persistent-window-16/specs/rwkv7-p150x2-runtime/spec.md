# RWKV-7 P150x2 Runtime Delta

## ADDED Requirements

### Requirement: Bounded persistent decode window-16 evaluation
r[tenstorrent.rwkv7_p150x2.persistent_decode_window_16] The P150x2 runtime SHALL evaluate persistent decode window `16` through a profile-bound, finite Metal Trace candidate before production admission.

#### Scenario: Candidate evaluation remains separate from production
- GIVEN the accepted production profile admits windows `2`, `4`, and `8`
- WHEN an operator runs the window-`16` candidate without explicit candidate control
- THEN production dispatch does not select persistent window `16`
- AND unsupported production requests use normal dispatch with `window_not_admitted`

#### Scenario: Window 16 passes complete evidence
- GIVEN the immutable candidate, free devices `0` and `1`, and bounded evidence budgets
- WHEN simulator, lifecycle, profiler, A/B, and canary validation complete
- THEN every accepted result has exact generated-token and FP32 recurrent-state parity
- AND every request is acknowledged, zeroed, and released
- AND each one-card and two-card paired speedup is at least `1.0`
- AND aggregate monitoring has no warning or critical alert and recommends `retain`

#### Scenario: Window 16 becomes admitted
- GIVEN all required evidence passes and binds the same source, profile, workload, runtime, and policy identities
- WHEN the production profile changes
- THEN its ordered admitted window set is `[2, 4, 8, 16]`
- AND window `16` uses one finite trace replay with zero per-token host submissions
- AND windows `2`, `4`, and `8` remain admitted

#### Scenario: An evidence gate fails
- GIVEN a profile error, fallback error, parity mismatch, cleanup error, profiler drop, speed loss, timeout, canary error, or monitoring alert
- WHEN the evidence is classified
- THEN production admission remains `[2, 4, 8]`
- AND no receipt claims window-`16` production admission
