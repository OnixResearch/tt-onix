## Implementation tasks

- [x] Run the coordinator baseline and define slot generations, deadlines, queue limits, and refill decisions in a pure core. [task:s1] [serial] r[tenstorrent.qwen38_p150x2.continuous_slot_policy]
- [x] Add staggered-arrival, unequal-length, cancellation, stale-event, capacity, and fairness tests. [task:s2] [after:s1] r[tenstorrent.qwen38_p150x2.continuous_slot_policy]
- [x] Route all device work through one shell worker and define stable refill and inactive-row contracts. [task:s3] [after:s2] r[tenstorrent.qwen38_p150x2.continuous_slot_effects]
- [x] Prove fake-adapter ownership, valid refill, state isolation, drift rejection, and timeout quarantine, then rerun the baseline. [task:s4] [after:s3] r[tenstorrent.qwen38_p150x2.continuous_slot_effects]
- [ ] Evaluate representative simulator refill or preserve its exact blocker before an approved physical B4 screen. [task:s5] [after:s4] r[tenstorrent.qwen38_p150x2.continuous_slot_admission]
- [ ] Compare queue-inclusive mixed-length latency, fairness, memory, and throughput against the current coordinator. [task:s6] [after:s5] r[tenstorrent.qwen38_p150x2.continuous_slot_admission]
