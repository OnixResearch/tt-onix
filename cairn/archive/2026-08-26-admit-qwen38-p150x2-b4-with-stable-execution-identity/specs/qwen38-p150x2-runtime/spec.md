## MODIFIED Requirements

### Requirement: Batched throughput admission uses matched physical roots

r[tenstorrent.qwen38_p150x2.batch_performance_admission]

The repository MUST define batch admission in a closed typed profile. The profile MUST compare the same traced production path at widths B=1 and B=4. Physical execution MUST use devices 0 and 1 in exact `single,batch,batch,single` root order. Each root MUST use the same prompt corpus, generation count, execution artifact, runtime, checkpoint, tokenizer, mesh, and immutable execution policy. The execution identity MUST bind all code, patches, runtime dependencies, and policy that can change measured behavior. It MUST exclude only receipt-binding metadata that cannot change execution. The final deployed closure MUST reference the measured execution identity before `physically_admitted` can become true. Admission MUST require exact per-request token parity, a complete active state prefix, distinct state slices, clean reset, aggregate decode speedup at least `3.0`, per-user decode-rate retention at least `0.75`, TTFT ratio at most `4.5`, and tail-latency ratio at most `1.5`. The existing single-stream admission profile and its `1.05` TTFT gate MUST remain unchanged.

#### Scenario: B=4 passes every batch gate

- GIVEN complete current-identity B=1 and B=4 physical roots with matching requests
- WHEN the deterministic batch receipt evaluator checks correctness, isolation, timing, throughput, cleanup, and execution identity
- THEN it MUST admit only when every profile-owned bound passes
- AND the admitted profile MUST bind the receipt BLAKE3 identity.

#### Scenario: Receipt metadata is promoted

- GIVEN a passing receipt for the immutable execution identity
- WHEN only the receipt hash and admission decision change
- THEN the execution identity MUST remain unchanged
- AND the final deployed closure MUST reference that same identity and receipt.

#### Scenario: An execution input changes

- GIVEN a measured execution identity
- WHEN code, patches, runtime dependencies, or immutable execution policy change
- THEN the execution identity MUST change
- AND the old receipt MUST remain ineligible for admission.
