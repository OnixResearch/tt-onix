## ADDED Requirements

### Requirement: Profile-selected persistent decode windows
r[tenstorrent.rwkv7_p150x2.persistent_decode_windows] The runtime MAY admit multiple finite persistent decode windows. The base profile MUST own a closed, ordered set of window lengths and a maximum trace bound. The runtime MUST derive request-local lifecycle limits only from a listed window, MUST preserve the base profile identity, and MUST retain normal dispatch for every unsupported window.

#### Scenario: Listed longer window passes
r[tenstorrent.rwkv7_p150x2.persistent_decode_windows.valid]
- GIVEN a homogeneous request whose decode length is listed in the physically admitted profile
- AND exact token, FP32 recurrent-state, profiler, simulator, cleanup, and stable physical-gain evidence exists for that length
- WHEN capability validation selects persistent decode
- THEN one bounded trace replay MUST execute the complete selected window
- AND the host MUST perform no per-token model submission inside that window.

#### Scenario: Conversational prompt runs
r[tenstorrent.rwkv7_p150x2.persistent_decode_windows.real_prompt]
- GIVEN a tokenizer-backed conversational prompt with a pinned FP32 CPU-reference continuation
- WHEN an admitted persistent window executes the prompt
- THEN generated tokens MUST exactly match normal dispatch and the pinned continuation
- AND the final FP32 recurrent-state BLAKE3 MUST exactly match normal dispatch.

#### Scenario: Window is not listed
r[tenstorrent.rwkv7_p150x2.persistent_decode_windows.unlisted]
- GIVEN a positive bounded decode length that is not in the profile window set
- WHEN capability validation runs
- THEN validation MUST reject trace capture before device execution
- AND the runtime MUST select normal dispatch with an explicit fallback reason.

#### Scenario: Cancellation arrives before replay
r[tenstorrent.rwkv7_p150x2.persistent_decode_windows.cancelled]
- GIVEN an admitted request whose cancellation is visible before trace replay
- WHEN the runtime processes the cancellation
- THEN the trace MUST NOT execute
- AND the lifecycle MUST acknowledge cancellation, zero recurrent state, and release the slot in that order.

#### Scenario: Longer window exceeds a bound
r[tenstorrent.rwkv7_p150x2.persistent_decode_windows.bound_exceeded]
- GIVEN a listed window whose trace exceeds its time, output, or trace-capacity bound
- WHEN runtime validation or execution detects the excess
- THEN the window MUST fail closed without production admission
- AND the accepted shorter window MUST remain available.

#### Scenario: Longer window has no stable gain
r[tenstorrent.rwkv7_p150x2.persistent_decode_windows.no_gain]
- GIVEN a parity-correct longer window without stable one-card and two-card physical gain
- WHEN production admission evaluates its order-balanced evidence
- THEN the longer window MUST remain unadmitted
- AND the runtime MUST NOT generalize the shorter-window performance claim.
