# Search frame

Goal: determine whether low-overhead signposts make host enqueue duplicates comparable.

Completion: package and self-tests pass; four physical roots use exact `eager,candidate,candidate,eager`; receipt either validates attribution or returns an exact blocker; production is restored.

False completions: Python profiling, missing signposts, unstable duplicates, simulator evidence, and profiler-overhead timing cannot support attribution.

Budget: one four-root P150x2 pilot, one warm-up and one measured request per root, no subagents.

Surviving mechanism: Tracy messages provide boundary timestamps while existing C++ zones provide enqueue and finish durations. Adversarial gate: every mode must keep wall, enqueue, and gap duplicate ratios at or below `1.25`.
