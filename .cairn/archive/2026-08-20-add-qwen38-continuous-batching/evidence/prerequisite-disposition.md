# Continuous-batching prerequisite disposition

Date: 2026-08-20

## Verdict

`UNSUPPORTED_UNADMITTED_EXECUTION_BASE`

The current Qwen3.8-27B P150x2 service does not support continuous batching.
Request execution remains serialized and eager.

## Prerequisite evidence

The proposed design requires traced decode for its batched runtime shell.
The physical traced-decode receipt has these results:

- Classification: `rejected`
- Admission: `false`
- Rejection reason: `ttft-regression-above-bound`
- Observed TTFT ratio: `1.2164174383938555`
- Maximum TTFT ratio: `1.05`
- Exact token-ID parity: `true`
- Clean reset: `true`
- Receipt BLAKE3: `33c3654bf417f5bf671cb127977d12465abbab2d6d524117d8d32f8b4cd11a37`
- Receipt path: `pkgs/qwen36/evidence/2026-08-20/qwen38-27b-traced-decode-candidate-receipt.json`

The production service reported `active` and returned `{"model": "Qwen3.8-27B", "status": "ok"}` after the physical run.

## Decision

No scheduler, queue, batched state, mixed-lane mask, or batched trace path is added.
A new change can reconsider batching after it names an admitted execution base.
That change must collect batch-specific parity, throughput, TTFT, token-latency, fairness, memory, and cleanup evidence.

## Non-claims

- The traced candidate's decode rate does not prove aggregate batch throughput.
- This disposition does not show that continuous batching is technically impossible.
- No concurrent-request performance result is claimed.
- No product or service behavior changed.
