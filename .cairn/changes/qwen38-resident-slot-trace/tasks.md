## Implementation tasks

- [x] Keep one stable slot decode trace across groups of the same width and route, releasing it on drift, a route change, or an eager next refill, with device-free adapter and planner probes. [task:t1] [serial] r[tenstorrent.qwen38_p150x2.resident_slot_trace]
- [x] Log each request's resumed and saved prefix and trace captures, and each group's trace decision. [task:t2] [after:t1] r[tenstorrent.qwen38_p150x2.resident_slot_trace]
- [x] Predeclare the ABBA admission policy, evaluator and receipt against the deployed package, with seeded sampled turns in the screen driver. [task:t3] [after:t2] r[tenstorrent.qwen38_p150x2.resident_slot_trace]
- [x] Measure baseline and candidate roots in ABBA order on hardware, re-screen the prefix cache on the candidate package, and record both bounded verdicts. [task:t4] [after:t3] r[tenstorrent.qwen38_p150x2.resident_slot_trace]
- [x] Deploy the candidate package only if both receipts admit it, and verify the served evidence. [task:t5] [after:t4] r[tenstorrent.qwen38_p150x2.resident_slot_trace]
