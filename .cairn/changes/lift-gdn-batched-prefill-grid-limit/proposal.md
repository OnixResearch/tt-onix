## Why

The B=8 batch-admission candidate was rejected on 2026-08-26 on a near miss.
It measured 99.29 aggregate tokens/second against 16.88 at B=1, but three
predeclared bounds failed:

| Gate | Measured | Bound | Result |
| --- | ---: | ---: | --- |
| Aggregate decode speedup | 5.8815 | at least 6.0 | FAIL |
| Per-user decode retention | 0.7352 | at least 0.75 | FAIL |
| TTFT ratio | 9.0792 | at most 9.0 | FAIL |

The rejection named its own root cause: eight serialized per-user prefills
dominate each request, and the extra four users add decode-step time without a
proportional decode gain. The receipt explicitly did not measure B=8 under a
runtime with faster or batched prefill.

That runtime limit is known and bounded. The GDN chunk-prefill program factory
refuses any batched prefill where `BH` exceeds the compute-core count, with
`BH = B x Nv_tp`. At TP=2 with 48 value heads the B=8 case gives
`8 x 24 = 192` against 110 compute cores. Prompts above 256 tokens avoid the
wall only because they prefill per user, which is why long-context B=8 works
and short-context B=8 needs a fallback.

Upstream PR 53482 already carries the fix as its second hardening item. It
lifts `TT_FATAL(BH <= grid.x * grid.y)` and assigns each active core a
contiguous chunk of `ceil(BH / ncores)` instances, with `(bh_start, n_inst)`
passed as runtime arguments. The mapping is unchanged when `BH` is at or below
the core count.

This change vendors that grid fix on its own, with its own parity evidence. It
does not admit any batch width.

## What Changes

- Vendor the `BH > ncores` support from the upstream op hardening into the
  package overlay, keeping the existing mapping for `BH <= ncores`.
- Add a pure, device-free instance-assignment core that computes contiguous
  per-core chunks and rejects malformed ranges.
- Add batched-prefill parity evidence: exact per-position parity against the
  serialized per-user prefill reference for the GDN chunk path.
- Keep the existing `QWEN_BATCHED_GROUPED` fallback reachable and fail-closed.

This change improves a prefill mechanism only. It admits no batch width, no
timing claim, and no serving route.

## Impact

- **Files**: `pkgs/qwen36/` (op overlay patch, instance-assignment core, check
  probes), this change directory, and the physical evidence tree.
- **Testing**: qwen36 device-free check with positive and negative assignment
  probes, package build, a fresh Blackhole simulator smoke, and matched
  physical prefill parity roots.
- **Production**: unchanged. The batched path stays behind its existing gates
  and no admission binding moves.
