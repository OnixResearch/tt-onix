## Phase 1: Measured full-model batch curve

- [x] [serial] Measure the full-model per-step time at active batch B in (1,4,8,16) on P150x2. Fix the two known integration points: `Qwen36Model.reset_state` forwarding an argument into the argless GDN `reset_state`, and the production `_forward_decode` page-table/position tensor staging. Record the full-model curve and the aggregate tokens/s implied, for r[tenstorrent.qwen38_p150x2.batched_concurrency].
- [x] [serial] Confirm the transfer ratio: full-model step growth per batch must be well below linear (layer-only evidence is flat) or the concurrency win is bounded and must be re-scoped, for r[tenstorrent.qwen38_p150x2.batched_concurrency].

## Phase 2: Pure batch-scheduling core

- [x] [serial] Add a pure, device-free batch-scheduling core: given active users with positions and per-user state slices, select the active batch width (1..16), advance each user exactly one token, and name each user's state slice. Reject malformed or out-of-range selections, for r[tenstorrent.qwen38_p150x2.batched_concurrency].
- [x] [serial] Wire positive and negative packaged-check probes for the scheduling core (valid batch, mixed users with isolated slices, width outside 1..16, state slice collision), for r[tenstorrent.qwen38_p150x2.batched_concurrency].

## Phase 3: Service integration + matched evidence

- [x] [serial] Integrate the scheduler into the service decode shell: one full-model decode at the active width per step, per-user token output, per-user recurrent/KV state preserved across steps, for r[tenstorrent.qwen38_p150x2.batched_concurrency].
- [x] [serial] Run a matched concurrent-versus-single-stream physical pair in one authorized window and record the aggregate served tokens-per-second delta versus the ~17 t/s single-stream floor, for r[tenstorrent.qwen38_p150x2.batched_concurrency].
- [x] [serial] Preserve receipts, sync the requirement, and archive without an admission claim, for r[tenstorrent.qwen38_p150x2.batched_concurrency].
