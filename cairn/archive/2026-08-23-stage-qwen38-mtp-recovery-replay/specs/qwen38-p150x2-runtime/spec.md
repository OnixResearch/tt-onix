# Qwen3.8 P150x2 Runtime Delta

## ADDED Requirements

### Requirement: Staged MTP recovery replay preserves the last completed boundary

r[tenstorrent.qwen38_p150x2.mtp_recovery_stage_probe]

The diagnostic P150x2 command MUST run one configured prompt and recovery route through the unchanged recovery path. It MUST atomically preserve ordered progress before and after trace, replay, recovery, verification, and cleanup boundaries. Unknown, repeated, time-regressing, or route-incompatible stages MUST fail closed. The staged result MUST remain admission-ineligible and MUST NOT change eager production behavior.

#### Scenario: Staged replay completes

GIVEN a matched P150x2 package, runtime, checkpoint, mesh, prompt, and recovery route
WHEN the staged command completes within its named bound
THEN the receipt and progress ledger identify every completed boundary and preserve exact diagnostic correctness fields

#### Scenario: Staged replay reaches its bound

GIVEN a matched staged command that does not complete
WHEN the external watchdog terminates the command
THEN the atomic progress ledger preserves the last completed boundary and eager production health is restored
