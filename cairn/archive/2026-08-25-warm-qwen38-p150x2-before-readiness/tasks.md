## Phase 1: Readiness policy and shell

- [x] [serial] Add a pure startup warm-up plan and readiness lifecycle with positive and negative device-free tests, for r[tenstorrent.qwen38_p150x2.startup_readiness].
- [x] [serial] Add batch-shell warm-up for B=1 and B=4 prefill, trace capture, one replay, and clean reset, for r[tenstorrent.qwen38_p150x2.startup_readiness].
- [x] [serial] Keep the HTTP socket unavailable until warm-up passes. Expose warm-up facts in `/healthz` and `/readyz`, for r[tenstorrent.qwen38_p150x2.startup_readiness].
- [x] [serial] Make systemd use notification readiness and fail closed when notification fails, for r[tenstorrent.qwen38_p150x2.startup_readiness].

## Phase 2: Validation and rollout

- [x] [serial] Run the package and repository checks, for r[tenstorrent.qwen38_p150x2.startup_readiness].
- [x] [serial] Deploy one physical restart and prove `warmup-passed` occurs before `ready`, for r[tenstorrent.qwen38_p150x2.startup_readiness].
- [x] [serial] Run bounded B=1/B=4/B=8/B=16 requests and negative streaming, timeout, cancellation, and restart checks. Preserve state-isolation evidence, for r[tenstorrent.qwen38_p150x2.startup_readiness].
- [x] [serial] Sync the accepted requirement and archive without an admission claim, for r[tenstorrent.qwen38_p150x2.startup_readiness].
