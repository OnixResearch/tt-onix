## Phase 1: Policy and evidence core

- [ ] [serial] Add the closed Nickel batch-admission profile and positive and negative export checks for r[tenstorrent.qwen38_p150x2.batch_performance_admission].
- [ ] [serial] Add the deterministic batch receipt evaluator with positive and negative fixtures for r[tenstorrent.qwen38_p150x2.batch_performance_admission].

## Phase 2: Service evidence

- [ ] [serial] Correct batch TTFT boundaries and add explicit operator-only admission evidence for r[tenstorrent.qwen38_p150x2.batch_ttft_evidence].
- [ ] [serial] Add device-free tests for eligible mode, normal-mode rejection, phase separation, identity drift, token mismatch, state collision, and cleanup failure for r[tenstorrent.qwen38_p150x2.batch_ttft_evidence].

## Phase 3: Physical decision

- [ ] [serial] Run matched physical roots in `single,batch,batch,single` order and evaluate every gate for r[tenstorrent.qwen38_p150x2.batch_performance_admission].
- [ ] [serial] Bind a passing receipt or preserve the exact rejection without weakening any bound for r[tenstorrent.qwen38_p150x2.batch_performance_admission].

## Phase 4: Lifecycle completion

- [ ] [serial] Restore production health, validate Cairn and package gates, sync the accepted spec, archive the change, and integrate it through `origin/main` for r[tenstorrent.qwen38_p150x2.batch_performance_admission].
