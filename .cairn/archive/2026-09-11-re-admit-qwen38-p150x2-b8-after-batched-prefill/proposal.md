## Why

The B=8 batch-admission candidate was rejected on 2026-08-26. Every correctness
gate passed. Three performance gates failed by narrow margins:

| Gate | Measured | Bound | Result |
| --- | ---: | ---: | --- |
| Aggregate decode speedup | 5.8815 | at least 6.0 | FAIL |
| Per-user decode retention | 0.7352 | at least 0.75 | FAIL |
| TTFT ratio | 9.0792 | at most 9.0 | FAIL |
| Tail-latency ratio | 1.4462 | at most 1.5 | pass |

Served B=8 throughput was real: 99.29 aggregate tokens/second against 16.88 at
B=1. The rejection withheld the claim, not the capability.

The receipt also recorded an explicit non-claim: it did not measure B=8 under a
runtime with faster or batched prefill. The rejection root cause names eight
serialized per-user prefills as the dominant cost.

The sibling change `lift-gdn-grid-instance-cap` removes the factory
guard that forces those serialized prefills. A fresh matched run is therefore
the only way to decide whether the near miss was a prefill-scheduling artifact
or a property of the served path.

This change re-runs the existing profile. It does not author a new profile, and
it does not relax a bound.

## What Changes

- Re-run the existing closed B=8 batch-admission profile against the rebuilt
  package that carries the batched-prefill mechanism.
- Keep every predeclared bound unchanged: aggregate speedup, per-user
  retention, TTFT ratio, tail-latency ratio, exact token parity, exact text
  parity, state slices, clean reset, and identity match.
- Keep the B=4 profile, contracts, binding, and receipt untouched.
- Run matched physical roots in exact `single,batch,batch,single` order on
  devices 0 and 1 over the existing eight-prompt corpus.
- Record an exact receipt for the verdict. Move `physically_admitted` to true
  only when every bound passes.

B=32 stays outside this change. It fails on a matmul divisibility assert at
TP=2 and is not a bound question.

## Impact

- **Files**: `pkgs/qwen36/` evidence entry point and binding, this change
  directory, and the physical evidence tree.
- **Testing**: Package checks, profile export determinism, positive and
  negative receipt self-tests, Cairn gates, and matched physical roots.
- **Production**: the B=8 binding moves only on a fully passing receipt. A
  rejection leaves the B=4 admission as the only admitted width.
