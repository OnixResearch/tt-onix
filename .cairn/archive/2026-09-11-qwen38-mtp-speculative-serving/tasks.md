# Tasks

- [x] [serial] Complete the pure round policy and measured economics transition. Keep device effects outside the core. r[tenstorrent.qwen38_p150x2.mtp_serving_rounds.policy]
- [x] [serial] Run positive and negative checks for scheduling, commit, economics, and request cancellation. r[tenstorrent.qwen38_p150x2.mtp_serving_rounds.valid] r[tenstorrent.qwen38_p150x2.mtp_serving_rounds.invalid]
- [x] [serial] Bind the serving module into the package identity and pass the baseline device-free check. r[tenstorrent.qwen38_p150x2.mtp_serving_rounds]
- [x] [serial] Complete the serving adapter with persistent traces, exact rollback, plain fallback, and cleanup. Closed by the 2026-09-11 owner decision: the bitwise restoration oracle is unattainable on tt-metal 0.77 (negative zero does not survive first device residency) and the path stays gated off; see `evidence/owner-decision-20260911.md`. r[tenstorrent.qwen38_p150x2.mtp_serving_rounds]
- [x] [serial] Complete the default-off server gate, module option, readiness field, and round metrics. Preserve other routes. r[tenstorrent.qwen38_p150x2.mtp_serving_rounds]
- [x] [serial] Run the physical greedy parity screen against plain decode for 64 tokens. Require exact parity; on divergence record the first index and the minimum verification top-two margin to classify a near-tie. Measure capture and round costs separately. r[tenstorrent.qwen38_p150x2.mtp_serving_rounds.valid]
- [x] [serial] Obtain owner review of the measured operating point, disengagement policy, and enablement. Recorded 2026-09-11: owner delegated the outcome review; keep gated off, archive as not admitted, do not weaken the bitwise oracle; see `evidence/owner-decision-20260911.md`. r[tenstorrent.qwen38_p150x2.mtp_serving_rounds]

## Baseline

The checked tasks record earlier implementation and bounded screens, not production acceptance. Cache regeneration restored the pilot golden, but the exact cache fault remains unknown. Later measurements and corrected counts are in `docs/qwen38-mtp-serving-progress.md`.

The current implementation also leaves a gap against the rejected-KV restoration requirement: it relies on masks and rewrite-before-read. Task 4 is open again for that gap. Task 7 stays open. These bounded results do not permit production admission, spec sync, or archive.

2026-09-11: the negative-zero blocker is root-caused at the platform level. Simulator and silicon agree that a negative zero is not observable on device after host conversion, so the bitwise restoration oracle is unattainable on this tt-metal 0.77 runtime by any rollback implementation. Task 4 now waits on the task 7 owner decision: normalize zero signs on both sides of the restoration comparison, or keep the path gated off. See `docs/qwen38-mtp-serving-progress.md`.

2026-09-11: owner decision recorded in `evidence/owner-decision-20260911.md` — keep gated off, archive as not admitted. Tasks 4 and 7 are closed by that decision.
