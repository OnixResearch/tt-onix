## Historical completed work

These markers retain the prior task record. They do not establish that the expanded September acceptance contract passes.

- [x] Implement the initial pure round schedule and greedy commit rule in `mtp_serving_core.py`. [task:m1] [serial] r[tenstorrent.qwen38_p150x2.mtp_serving_rounds.policy]
- [x] Add the initial positive and negative policy probes to `pkgs/qwen36/check.py`. [task:m2] [after:m1] r[tenstorrent.qwen38_p150x2.mtp_serving_rounds.valid]
- [x] Bind the initial module into the package execution identity and record its prior device-free result. [task:m3] [after:m2] r[tenstorrent.qwen38_p150x2.mtp_serving_rounds]

## Remaining implementation and validation

- [ ] Run the current baseline and fix the undefined success timer, loss-streak transition, complete timing, and route-bound calibration. [task:m4] [after:m3] r[tenstorrent.qwen38_p150x2.mtp_serving_economics]
- [ ] Define every persistent allocation and repair current-round feedback, worst-case bounds, trace coexistence, and plain fallback. [task:m5] [after:m4] r[tenstorrent.qwen38_p150x2.mtp_serving_lifecycle]
- [ ] Move watchdog effects into the shell and enforce one device owner across MTP, sampled, and ordinary decode routes. [task:m6] [after:m5] r[tenstorrent.qwen38_p150x2.mtp_serving_lifecycle]
- [ ] Add exact multi-round fixtures for all acceptance counts, accept-then-zero-reject, repeated zero acceptance, and successful request return. [task:m7] [after:m6] r[tenstorrent.qwen38_p150x2.mtp_serving_rounds.valid]
- [ ] Add timeout-with-live-worker, deadline, concurrent-route, partial-allocation, stale-binding, and failed-cleanup fixtures, then rerun the baseline. [task:m8] [after:m7] r[tenstorrent.qwen38_p150x2.mtp_serving_rounds.invalid]
- [ ] Complete default-off module wiring and readiness fields without changing off-route batch or sampled behavior. [task:m9] [after:m8] r[tenstorrent.qwen38_p150x2.mtp_serving_rounds]
- [ ] Run representative simulator state carry or retain its exact blocker before an explicitly authorized physical screen. [task:m10] [after:m9] r[tenstorrent.qwen38_p150x2.mtp_serving_rounds.valid]
- [ ] Compare repeated greedy requests with the plain control, including exact hybrid state, every rejection position, and complete economic timing. [task:m11] [after:m10] r[tenstorrent.qwen38_p150x2.mtp_serving_economics]
- [ ] Record owner review of the operating point and enable only the exact admitted identity, or retain the rejected default-off result. [task:m12] [after:m11] r[tenstorrent.qwen38_p150x2.mtp_serving_rounds]
