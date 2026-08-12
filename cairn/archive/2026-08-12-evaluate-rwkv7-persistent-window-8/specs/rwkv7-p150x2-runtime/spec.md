# RWKV-7 P150x2 Runtime Delta

## ADDED Requirements

### Requirement: Bounded persistent decode window-8 evaluation
r[tenstorrent.rwkv7_p150x2.persistent_decode_window_8] The P150x2 runtime SHALL evaluate persistent decode window `8` through a profile-bound, finite Metal Trace candidate before it adds that window to production admission.

#### Scenario: Candidate evaluation remains separate from production
- GIVEN the accepted production profile admits only windows `2` and `4`
- WHEN an operator builds or runs the window-`8` candidate without an explicit candidate control
- THEN production dispatch does not select persistent window `8`
- AND unsupported production requests use normal dispatch with `window_not_admitted`

#### Scenario: Window 8 passes complete evidence
- GIVEN the immutable window-`8` candidate, free physical devices `0` and `1`, and the bounded evidence budgets
- WHEN simulator, lifecycle, profiler, order-balanced A/B, and canary validation complete
- THEN every accepted persistent result has exact generated-token and FP32 recurrent-state parity
- AND every request is acknowledged, zeroed, and released without timeout or terminal failure
- AND one-card and two-card minimum paired speedup are each at least `1.0`
- AND aggregate monitoring has no warning or critical alert and recommends `retain`

#### Scenario: Window 8 becomes admitted
- GIVEN all required window-`8` evidence passes and binds the same source, profile, workload, runtime, and policy identities
- WHEN the production profile is updated
- THEN its ordered admitted window set is `[2, 4, 8]`
- AND window `8` uses one finite trace replay with zero per-token host submissions
- AND windows `2` and `4` remain admitted

#### Scenario: Any evidence gate fails
- GIVEN a malformed profile, unsupported window, parity mismatch, cleanup failure, profiler drop, non-positive paired gain, timeout, terminal failure, invalid canary, or monitoring alert
- WHEN window-`8` evidence is classified
- THEN production admission remains `[2, 4]`
- AND no receipt claims window-`8` production admission
