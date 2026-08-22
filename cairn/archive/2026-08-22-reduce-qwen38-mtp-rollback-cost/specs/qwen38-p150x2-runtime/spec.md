## ADDED Requirements

### Requirement: MTP rejection uses bounded device-resident rollback evidence

r[tenstorrent.qwen38_p150x2.mtp_kv_rollback] The repository MAY provide a diagnostic P150x2 control that compares MTP cache re-prefill with device-resident rollback. The control MUST use pinned identities, physical devices `0` and `1`, batch one, exactly three drafts, the bounded shard-local selector, and at least five fixed prompts. It MUST retain at least five complete samples per route and prompt in balanced `reprefill,snapshot,snapshot,reprefill` blocks. Before draft trace capture, the snapshot route MUST preserve only the MTP K/V rows that fixed-shape draft decode can overwrite and MUST preserve prefix feedback in preallocated device buffers. After target verification, it MUST restore every rejected row, retain every accepted row, process any accepted draft not already represented in feedback, and advance exactly one committed target token. It MUST compare the next draft token, feedback tensor, and committed MTP K/V prefix with re-prefill. The evaluator MUST reject missing or unknown evidence, route drift, invalid duration, identity drift, state mismatch, acceptance mismatch, duplicate-root recovery spread above `1.25`, or snapshot/re-prefill recovery ratio above the configured bound. A valid result MAY select snapshot rollback for later diagnostic MTP work. It MUST NOT admit MTP, traced target decode, reduced precision, or production behavior.

#### Scenario: Snapshot rollback is selected

- **GIVEN** matched re-prefill and snapshot roots for all fixed prompts
- **WHEN** next-round MTP state matches exactly, recovery roots are stable, and snapshot recovery is materially faster
- **THEN** the evaluator SHALL select snapshot rollback for later diagnostic MTP work while production remains eager

#### Scenario: Rollback evidence fails closed

- **GIVEN** missing evidence, route drift, unstable timing, invalid timing, state mismatch, or insufficient recovery gain
- **WHEN** the deterministic evaluator runs
- **THEN** it MUST reject snapshot rollback and MUST select no production change
