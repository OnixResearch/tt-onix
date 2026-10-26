## ADDED Requirements

### Requirement: Traffic and retention tools bind assumptions and measurement provenance

r[tenstorrent.tooling.traffic_retention]

The packaged CLI MUST label traffic as analytic modeled tile/payload/exponent/L1 accounting, not measured counters. Retention scoring MUST bind the frozen original passages and tokenizer/config identities, score targets from index 13 through the end over exactly 248320 vocabulary columns, cast to float32 before float64 log-softmax, and equally weight six passage means. Hardware collection MUST bind actual indexed weight-shard contents and immutable package/runtime identities; comparisons MUST reject undeclared provenance changes and MUST NOT imply serving admission.

#### Scenario: Traffic assumes padded tiles and multicast geometry

- GIVEN logical matrix dimensions, dtypes, block sizes and a compatible grid
- WHEN traffic is calculated
- THEN padding, exponent storage, rereads, intermediate materialization and L1 buffer copies are explicit, with no hardware throughput prediction.

#### Scenario: Full-vocabulary exports are compared

- GIVEN content-bound baseline and treatment exports with exact target-index rows
- WHEN installed score and compare commands execute
- THEN extra/truncated vocabulary, malformed rows, nonfinite values and unmatched provenance are rejected; the optional +0.010-nat result is limited to the original six-passage diagnostic.

#### Scenario: Model files change behind an unchanged path and config

- GIVEN an indexed checkpoint with unchanged config/tokenizer and changed referenced shard bytes
- WHEN collection or comparison checks content identity
- THEN the changed manifest is rejected rather than accepted under the declared checkpoint label.
