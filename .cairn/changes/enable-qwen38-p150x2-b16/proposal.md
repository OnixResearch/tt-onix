## Why

`MAXIMUM_ACTIVE_BATCH_WIDTH = 16` is the service's declared batch capacity, the
model allocates its decode buffers at that width, and no B=16 run has ever been
attempted. The width was unreachable for a non-obvious reason: the batch
admission policy is a closed set,

```python
ALLOWED_CANDIDATE_WIDTHS = (CANDIDATE_WIDTH, B8_CANDIDATE_WIDTH)   # (4, 8)
```

so any B=16 profile fails to load with `workload widths are unsupported` before
a device is touched. Nothing in the runtime capped width 16; the policy did.

A first attempt confirms both halves: with the closed set widened, the width-16
admission server starts, warms up, and serves sixteen concurrent requests at
`batch_width: 16`, and one full batch root completed at roughly 170 aggregate
tokens per second (against roughly 100 at width 8). An intermittent program
error is the remaining blocker for calling it reliable.

## What Changes

- Widen the closed batch-admission width set to include width 16, with the same
  four bounds the other widths use.
- Add the B=16 profile family (policy, contracts, profile, binding, validated
  variants) mirroring B=8, with a sixteen-prompt corpus and a
  `qwen38-p150x2-b16-admission` binary.
- Add device-free probes for the B=16 profile: width, corpus size, pinned
  bounds, measurement eligibility at width 16 and width 1, and the rejected
  mappings.
- Run the matched four-root protocol at width 16 and record either a passing
  receipt or the exact failing gate, plus the intermittent program error.

## Impact

- **Files**: `pkgs/qwen36/` policy, contracts, profile, corpus, probes, and the
  package's export and binary wiring; this change directory.
- **Testing**: the device-free check, the package build and install assertions,
  and the matched physical protocol at width 16.
- **Production**: production keeps serving the single-stream candidate route.
  This change adds a width; it does not enable batched serving, and it does not
  move the B=4 or B=8 bindings.
