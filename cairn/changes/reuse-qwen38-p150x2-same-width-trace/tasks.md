## Phase 1: Policy and evidence

- [ ] [serial] Add the closed Nickel reuse profile and pure transition policy for r[tenstorrent.qwen38_p150x2.same_width_trace_reuse].
- [ ] [serial] Add a deterministic Rust receipt evaluator with positive and negative self-tests for r[tenstorrent.qwen38_p150x2.same_width_trace_reuse].

## Phase 2: Service candidate

- [ ] [serial] Add one-trace same-width reuse to the adapter and keep normal production disabled for r[tenstorrent.qwen38_p150x2.same_width_trace_reuse].
- [ ] [serial] Record trace reuse, resident count, capture time, reset state, and request wall time for r[tenstorrent.qwen38_p150x2.same_width_trace_reuse].
- [ ] [serial] Add positive and negative device-free tests for width drift, binding drift, reset failure, multiple traces, and disabled policy for r[tenstorrent.qwen38_p150x2.same_width_trace_reuse].

## Phase 3: Physical decision

- [ ] [serial] Run matched B=4 roots in `recapture,reuse,reuse,recapture` order on devices `0` and `1` for r[tenstorrent.qwen38_p150x2.same_width_trace_reuse].
- [ ] [serial] Bind a passing receipt or preserve the exact rejection without weakening any bound for r[tenstorrent.qwen38_p150x2.same_width_trace_reuse].
- [ ] [serial] If reuse passes, rerun the existing B=1 versus B=4 admission gate on the new execution identity for r[tenstorrent.qwen38_p150x2.same_width_trace_reuse].

## Phase 4: Lifecycle completion

- [ ] [serial] Restore production health and validate package, Cairn, formatting, and repository gates for r[tenstorrent.qwen38_p150x2.same_width_trace_reuse].
- [ ] [serial] Sync, archive, and commit the validated result on the isolated branch for r[tenstorrent.qwen38_p150x2.same_width_trace_reuse].
