## Why

The proposed continuous-batching path requires an admitted traced-decode base.
The physical traced-decode candidate failed its TTFT gate and stayed unadmitted.

Building a batched trace on a rejected base would bypass the accepted performance and lifecycle boundary.

## Outcome

Return continuous batching as unsupported for the current Qwen3.8-27B P150x2 service.
The production service stays serialized and uses eager decode.

This result does not reject continuous batching on technical merit.
It rejects current support because the required traced-decode base is not admitted.

## What Changes

- Record the traced-decode rejection and its receipt identity.
- Keep the serialized eager service unchanged.
- Add no scheduler, queue, batch slot, batched state, or performance claim.
- Require a new validated Cairn change after its execution base is admitted.
- Require complete physical throughput, latency, parity, fairness, memory, and cleanup evidence for future support.

## Impact

- **Service**: request execution remains serialized.
- **Runtime**: no batched trace path is added.
- **Configuration**: no batch profile is added.
- **Evidence**: the archive records the rejected prerequisite.
- **Documentation**: no local continuous-batching claim is allowed.

## Out of Scope

- Reopening traced-decode admission.
- Estimating batch throughput from decode-rate results.
- Adding dormant scheduler or state-slot code.
- Changing model precision, runtime, or TP2 tuning.
