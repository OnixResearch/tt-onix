## ADDED Requirements

### Requirement: Gated exact greedy MTP serving

r[tenstorrent.qwen38_p150x2.mtp_serving_rounds]

The service MUST keep MTP default-off and leave off-route startup, sampling, batching, and admission unchanged. Enabled greedy rounds MUST verify the seed plus drafts and commit only the accepted prefix plus its continuation token. Hybrid state MUST match that committed prefix before another round. Production admission MUST require matched physical evidence and explicit owner review. The enabled route MUST use the content-bound BF16 MTP head. Disabled MTP MUST NOT load its weights or capture its traces.

#### Scenario: Fully accepted draft chain
- GIVEN a draft chain that matches every target continuation before the bonus
- WHEN the service commits the round
- THEN it emits the drafts plus the final continuation and retains the corresponding consumed-prefix state.

#### Scenario: Rejected draft chain
- GIVEN a mismatch at any draft position
- WHEN the service commits the round
- THEN it emits only the accepted prefix and rejection continuation, with no rejected state visible to later decode.

### Requirement: Pure round schedule and commit policy

r[tenstorrent.qwen38_p150x2.mtp_serving_rounds.policy]

The pure core MUST own scheduling, commit, loss-streak transitions, and disengagement. It MUST receive observed timing and calibration explicitly. It MUST NOT own threads, clocks, TTNN, or process termination. A speculative round MUST require room for all drafts plus a continuation. The verifier output MUST contain exactly one more row than the draft chain. The core MUST reject draft counts outside the existing closed range from 2 through 6.

#### Scenario: Eligible round and plain tail
- GIVEN valid counters and an admitted draft width
- WHEN the remaining token count permits a complete speculative round
- THEN the core selects speculation, otherwise it selects the bounded plain tail.

#### Scenario: Invalid policy input
- GIVEN an invalid width, negative counter, non-integer token, or incorrect continuation count
- WHEN policy evaluation runs
- THEN the core returns a specific rejection without an effect plan.

### Requirement: Positive multi-round evidence

r[tenstorrent.qwen38_p150x2.mtp_serving_rounds.valid]

The test suite MUST cover full acceptance and rejection at every draft position across successive rounds. It MUST compare target KV, recurrent state, convolution state, draft feedback, positions, and emitted tokens with a sequential reference. Prefix-zero feedback MUST refresh from the current round, not the initial prompt.

#### Scenario: Acceptance followed by zero acceptance
- GIVEN one accepted round followed by rejection of the next first draft
- WHEN the adapter restores the prefix
- THEN it restores the current round's feedback and target state exactly.

#### Scenario: Stale feedback fixture
- GIVEN feedback copied only from the initial prompt
- WHEN a later round rejects its first draft
- THEN the fixture detects the state mismatch and fails admission.

### Requirement: Negative lifecycle evidence

r[tenstorrent.qwen38_p150x2.mtp_serving_rounds.invalid]

The suite MUST reject invalid counters, mismatched rows, bad token IDs, stale trace bindings, partial allocation, and unsafe mode transitions. It MUST cover repeated zero acceptance, deadline expiry, and capture failure. An error MUST NOT permit continuation from unknown device state.

#### Scenario: Valid bounded cleanup
- GIVEN a completed worker and an explicit safe cleanup plan
- WHEN cleanup finishes
- THEN every owned trace and buffer releases once and the engine can accept another request.

#### Scenario: Abandoned worker
- GIVEN a watchdog timeout with a live device thread
- WHEN the shell handles the timeout
- THEN it blocks requests and ends the process without concurrent TTNN cleanup.

### Requirement: Stable allocation and exclusive ownership

r[tenstorrent.qwen38_p150x2.mtp_serving_lifecycle]

The adapter MUST establish the lifetime and overlap contract for every persistent buffer before replay. One shell worker MUST own MTP and all other device routes. Plain fallback MUST release or safely isolate resident traces first. Repeated requests MUST refresh prefix state without unbounded per-round allocation.

#### Scenario: Safe request transition
- GIVEN a completed MTP request and verified cleanup
- WHEN another greedy or sampled request arrives
- THEN the same owner starts it from isolated state with valid bindings.

#### Scenario: Concurrent route or unsafe allocation
- GIVEN another active device route or an unproved persistent allocation after capture
- WHEN the service prepares MTP
- THEN it defers or rejects the request before conflicting effects.

### Requirement: Complete MTP economics

r[tenstorrent.qwen38_p150x2.mtp_serving_economics]

The core MUST update the loss streak on every losing round and reset it only on a winning round. Calibration MUST bind the actual plain route. Round cost MUST include draft, verification, selection, commit, rollback, and synchronization. Request cost MUST also include prefill and capture. Failed timing stability MUST block admission.

#### Scenario: Consecutive economic losses
- GIVEN a predeclared loss threshold and complete observed round costs
- WHEN consecutive losses reach that threshold
- THEN the core requests sticky disengagement for the rest of the request.

#### Scenario: Omitted rollback or stale calibration
- GIVEN incomplete round timing or calibration from another route
- WHEN the evaluator considers MTP benefit
- THEN it rejects the comparison without a speedup claim.
