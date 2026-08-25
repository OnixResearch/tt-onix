## ADDED Requirements

### Requirement: P150x2 decode serves concurrently with matched evidence

r[tenstorrent.qwen38_p150x2.batched_concurrency]

The repository MAY batch multiple active decode streams on the P150x2 service so
aggregate served tokens-per-second exceeds the single-stream floor. Batching MUST
run one full-model decode step per active batch of width `B <= 16`. Per-user
recurrent and KV state MUST stay isolated across steps and across users. Batch
selection MUST live in a pure device-free core. Any claim that batching raises
served throughput MUST use a matched concurrent-versus-single-stream physical pair
under the existing decode admission discipline; a per-op or per-layer result alone
MUST NOT imply a served-throughput gain.

#### Scenario: Concurrent streams are served in one step

- GIVEN `K` active users with stable positions and per-user state slices
- WHEN the scheduler assembles an active batch of width `B <= 16` and runs one decode step
- THEN each user advances exactly one token with its own state preserved
- AND the aggregate served tokens-per-second is measured against the single-stream floor

#### Scenario: State isolation or scheduling fails closed

- GIVEN an active batch that mixes two users, or a scheduler that selects outside `1..16`
- WHEN the evaluator processes the step
- THEN the core MUST reject the selection with a bounded reason
- AND the service MUST keep the single-stream path selected until matched evidence exists
