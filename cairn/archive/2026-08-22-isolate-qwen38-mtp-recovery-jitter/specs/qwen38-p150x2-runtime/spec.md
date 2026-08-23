# Qwen3.8 P150x2 Runtime Delta

## ADDED Requirements

### Requirement: MTP recovery stability uses one persistent trace

r[tenstorrent.qwen38_p150x2.mtp_recovery_stability]

The diagnostic P150x2 control MUST use one fixed-shape MTP trace and stable buffers for draft and recovery replay in one loaded process. Selection MUST require one trace capture, exact binding reuse, exact next-round state, complete cleanup, six roots per route and prompt, recovery spread at most `1.25`, and a snapshot/re-prefill median ratio at most `0.8`. The control MUST remain diagnostic-only and MUST NOT change eager production behavior.

#### Scenario: Persistent recovery trace passes

GIVEN a matched P150x2 package, runtime, checkpoint, mesh, device set, prompt set, route schedule, and stable trace binding
WHEN one loaded process captures one MTP trace during discarded warmup and reuses it for all measured draft and recovery steps
THEN the receipt records one capture, exact binding reuse, exact next-round state, complete cleanup, stable route groups, and the bounded recovery ratio

#### Scenario: Persistent recovery trace fails closed

GIVEN a missing or repeated capture, changed binding, state mismatch, incomplete cleanup, unstable route group, or excessive recovery ratio
WHEN the deterministic evaluator reads the observations
THEN it rejects persistent recovery tracing and preserves eager production behavior
