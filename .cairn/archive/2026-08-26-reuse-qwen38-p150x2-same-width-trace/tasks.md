## Phase 1: Policy and evidence

- [x] [serial] Add the closed Nickel reuse profile and pure transition policy for r[tenstorrent.qwen38_p150x2.same_width_trace_reuse].
- [x] [serial] Add a deterministic Rust receipt evaluator with positive and negative self-tests for r[tenstorrent.qwen38_p150x2.same_width_trace_reuse].

## Phase 2: Service candidate

- [x] [serial] Add one-trace same-width reuse to the adapter and keep normal production disabled for r[tenstorrent.qwen38_p150x2.same_width_trace_reuse].
- [x] [serial] Record trace reuse, resident count, capture time, reset state, and request wall time for r[tenstorrent.qwen38_p150x2.same_width_trace_reuse].
- [x] [serial] Add positive and negative device-free tests for width drift, binding drift, reset failure, multiple traces, and disabled policy for r[tenstorrent.qwen38_p150x2.same_width_trace_reuse].

## Phase 3: Physical decision

- [x] [serial] Run matched B=4 roots in `recapture,reuse,reuse,recapture` order on devices `0` and `1` for r[tenstorrent.qwen38_p150x2.same_width_trace_reuse].
- [x] [serial] Bind a passing receipt or preserve the exact rejection without weakening any bound for r[tenstorrent.qwen38_p150x2.same_width_trace_reuse].
- [x] [serial] Record that the conditional B=1 versus B=4 rerun did not trigger because physical reuse was rejected for r[tenstorrent.qwen38_p150x2.same_width_trace_reuse].

## Phase 4: Lifecycle completion

- [x] [serial] Restore production health and validate package, Cairn, formatting, and repository gates for r[tenstorrent.qwen38_p150x2.same_width_trace_reuse].
- [x] [serial] Sync, archive, and commit the validated rejection on the isolated branch for r[tenstorrent.qwen38_p150x2.same_width_trace_reuse].
