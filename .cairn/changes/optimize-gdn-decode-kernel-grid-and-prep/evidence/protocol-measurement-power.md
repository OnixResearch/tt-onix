# Measurement integrity under the frozen B=8 protocol

Corrected 2026-09-13. The cause of the timing variation remains unresolved.
The original Dst candidate also violates destination capacity, independently of these timing results.
Further admission campaigns must not reuse that candidate.

## Keep the metric contract

`scripts/qwen36-batch-receipt.rs` requires `single,batch,batch,single` in both the command and profile.
Changing to ABAB, adding roots, or replacing the latency statistic changes the contract.
This work does not make those changes or loosen any admission bound.

`batch_service.py` records one TTFT value per group: the interval from prefill start to first-token readiness.
`batch_decode_adapter.py` observes readiness when the callback has logits for the whole width.
The service copies this group value into each response.
The receipt takes the median of response TTFT values and divides batch by single.
It does not take the maximum of eight independent first-token timestamps.
Those independent timestamps do not exist in the current response contract.

Steady decode rate uses the average measured decode duration, excluding the first step when possible.
Per-user rate is its reciprocal; aggregate rate multiplies that rate by width.
The receipt forms ratios from the corresponding medians.
Duplicated group fields are not independent observations for a statistical analysis.

## Round 4 contains a large root transition

The following values come from `00-measured-N.json` in each preserved round-4 batch root.
They describe three group samples per root, not 48 independent TTFT samples.

| Root | Run | TTFT (s) | State installation (s) | Trace preparation (s) | Per-user decode (tokens/s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| batch-1 | 0 | 11.191 | 1.045 | 0.991 | 12.691 |
| batch-1 | 1 | 11.194 | 1.044 | 0.951 | 12.716 |
| batch-1 | 2 | 11.458 | 1.216 | 1.000 | 12.147 |
| batch-2 | 0 | 69.094 | 11.924 | 6.146 | 11.555 |
| batch-2 | 1 | 71.280 | 10.956 | 4.765 | 11.423 |
| batch-2 | 2 | 75.538 | 7.784 | 4.311 | 11.680 |

The combined receipt reports median batch TTFT 40.276 seconds and median single TTFT 2.950 seconds.
The batch median lies between two widely separated root populations.
This is not merely a two-percent decision margin.
State installation and trace preparation also increase, but their cause is not isolated.
The original logs lack enough stage and host-load evidence to attribute the transition.

There were zero logged group failures in round 4.
That rules out counted group failures as a necessary condition for this rejected receipt.
It does not rule out host contention, cache differences, scheduling, runtime faults, or other unmeasured causes.
The first two curl rounds also completed without logged group failures.
The barrier client is therefore a mitigation attempt, not a proven fix.

## Proven service defect

The service previously evaluated the admission profile after prefill, trace preparation, decode, and reset.
An unsupported partial width could perform all those effects before policy rejection.
The worker then treated that policy rejection as an execution failure and reset again.

A device-free fake reproduced those unwanted calls before the fix.
`BatchAdmissionProfile.measurement()` now runs before the first adapter effect.
The worker reports a stable admission-stage rejection without resetting untouched state.
Execution failures still reset state and report `failure_stage="execution"`.
The change preserves production grouping and all existing profile decisions.

The historical logs do not prove that this policy rejection caused every old HTTP 500.
The new admission diagnostic identifies future occurrences without inferring a trace-capture failure.

## Remaining V9 work

Before another admission campaign:

1. Complete the bounded kernel's correctness and hazard checks.
2. Require HTTP success, complete evidence, exact width, and effective identities in both clients.
3. Preserve every attempt in a unique directory, including rejected groups and client errors.
4. Record root-level stage timings and host conditions before and after each root.
5. State the repetition count and failure rules before measurement. Do not discard failures based on their verdict.
6. Keep ABBA and the current latency statistic. Treat a proposed alternative as a separate, versioned protocol.
7. Require repeatable passing receipts, not a favorable subset of the recorded runs.

The earlier request to widen the protocol was based on an unproven causal explanation.
V9 now requires measurement integrity under the existing contract.
It remains open; these source reviews and host tests are not new admission evidence.
