# Tasks

- [x] [serial] Complete the pure round policy and measured economics transition. Keep device effects outside the core. r[tenstorrent.qwen38_p150x2.mtp_serving_rounds.policy]
- [x] [serial] Run positive and negative checks for scheduling, commit, economics, and request cancellation. r[tenstorrent.qwen38_p150x2.mtp_serving_rounds.valid] r[tenstorrent.qwen38_p150x2.mtp_serving_rounds.invalid]
- [x] [serial] Bind the serving module into the package identity and pass the baseline device-free check. r[tenstorrent.qwen38_p150x2.mtp_serving_rounds]
- [x] [serial] Complete the serving adapter with persistent traces, exact rollback, plain fallback, and cleanup. r[tenstorrent.qwen38_p150x2.mtp_serving_rounds]
- [x] [serial] Complete the default-off server gate, module option, readiness field, and round metrics. Preserve other routes. r[tenstorrent.qwen38_p150x2.mtp_serving_rounds]
- [x] [serial] Run the physical greedy parity screen against plain decode for 64 tokens. Require exact parity; on divergence record the first index and the minimum verification top-two margin to classify a near-tie. Measure capture and round costs separately. r[tenstorrent.qwen38_p150x2.mtp_serving_rounds.valid]
- [ ] [serial] Obtain owner review of the measured operating point, disengagement policy, and enablement. r[tenstorrent.qwen38_p150x2.mtp_serving_rounds]

## Baseline

The package and repository Cairn validation pass at `e205476`. Tasks 1 through 6 are complete with device-free suites and two physical screens. The root cause of zero acceptance was the 0.77 misreading of 0.74-written head caches; runtime-scoped cache directories restore the 0.74 acceptance level, and targeted per-row rollback cut round cost to 0.17 s with exact parity on every screen. Task 7 stays open: economics are near break-even, so do not enable without the owner's review of the remaining levers.
