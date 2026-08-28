# RWKV-7 P150x2 Runtime Delta

## ADDED Requirements

### Requirement: Persistent decode monitoring and alerts
r[tenstorrent.rwkv7_p150x2.persistent_decode_monitoring] The system MUST provide a deterministic bounded collector that validates persistent decode telemetry, emits aggregate production metrics, classifies fixed safety alerts, and recommends fail-closed persistent admission disablement for every critical result.

#### Scenario: Typed monitoring policy is valid
r[tenstorrent.rwkv7_p150x2.persistent_decode_monitoring.policy]
- GIVEN a strict policy with bounded event count and fallback thresholds
- WHEN Nickel exports the policy and Rust parses it
- THEN both layers accept the same policy values and fixed actions

#### Scenario: Valid receipt batch is collected
r[tenstorrent.rwkv7_p150x2.persistent_decode_monitoring.collect]
- GIVEN an explicit non-empty batch of valid worker receipts within the configured limit
- WHEN the monitoring core collects the batch
- THEN it emits bounded aggregates with ordered event BLAKE3 values and one deterministic batch BLAKE3
- AND it excludes paths, prompts, tokens, request identifiers, and binding identities

#### Scenario: Critical safety signal is observed
r[tenstorrent.rwkv7_p150x2.persistent_decode_monitoring.critical]
- GIVEN telemetry with parity failure, cleanup failure, timeout, terminal failure, unsupported persistent admission, malformed content, or duplicate event bytes
- WHEN the monitoring core classifies the batch
- THEN it emits the matching fixed critical alert
- AND it recommends persistent admission disablement

#### Scenario: Fallback rate exceeds policy
r[tenstorrent.rwkv7_p150x2.persistent_decode_monitoring.fallback]
- GIVEN enough accepted events to satisfy the configured minimum sample count
- AND the normal-dispatch fallback rate exceeds the configured maximum
- WHEN the monitoring core classifies the batch
- THEN it emits one fixed warning with the configured investigation action

#### Scenario: Operator runs the collector command
r[tenstorrent.rwkv7_p150x2.persistent_decode_monitoring.command]
- GIVEN a typed policy path and explicit worker receipt paths
- WHEN the operator runs the monitoring command
- THEN the command emits one monitoring receipt
- AND it returns a stable status for clean, warning, or critical results

#### Scenario: Monitoring input is invalid or unbounded
r[tenstorrent.rwkv7_p150x2.persistent_decode_monitoring.invalid]
- GIVEN an empty batch, oversized batch, invalid policy, non-finite threshold, unknown field, or unsupported schema
- WHEN the monitoring core validates the input
- THEN collection fails before it emits an admission recommendation
