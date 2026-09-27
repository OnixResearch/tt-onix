## Why

Agents send one turn at a time. The continuous-slot scheduler opens a new slot group for a turn that arrives alone, and every new group captures a new decode trace. The capture takes about 0.7–1.3 s between a turn's first and second token. After prefix-cache hits it is about a tenth of a typical agent turn.

The request log also omits the prefix and capture facts that each response carries. Nobody can tell from the log whether the cache serves real agent traffic.

## What Changes

- Keep the one resident slot decode trace across groups of the same width and route. Release it when its binding drifts, the route changes, or the next refill would prefill eagerly.
- Log each request's resumed prefix, saved prefix and trace captures in `request-passed`, and each group's trace decision in `continuous-slot-group-started`.
- Predeclare an ABBA comparison against the deployed package: exact parity, identical cache use, few captures, and faster requests without slower prefill.

## Impact

- **Files**:
  - `pkgs/qwen36/service/qwen36_service/batch_decode_core.py`: the pure `plan_slot_group_trace` decision.
  - `pkgs/qwen36/service/qwen36_service/batch_decode_adapter.py`: `begin_slots` applies the decision.
  - `pkgs/qwen36/service/qwen36_service/continuous_slot_service.py`: events.
  - `pkgs/qwen36/resident-trace-admission-*.ncl`, `resident_trace_admission_core.py`, `resident_trace_receipt.py`: the admission.
  - `scripts/qwen38-prefix-cache-screen.py`: seeded sampled workloads, and captures per turn.
- **Testing**:
  - Device-free probes `batch decode scheduler`, `continuous slot adapter` and `resident trace admission`, each mutation-checked.
  - A physical ABBA screen against the deployed package.
  - A re-screen of the prefix cache on the new package.
