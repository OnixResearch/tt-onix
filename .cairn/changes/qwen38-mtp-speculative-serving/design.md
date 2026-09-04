## Context and current state

The September research report records the retained diagnostic results and source defects. Serving code, a feedback pool, and watchdogs exist. End-to-end admission remains open.

The previous design contained two different verifier conventions. This design uses only the seed-plus-drafts convention implemented by the current commit core.

## Exact round model

Let draft count be `k`, accepted count be `j`, and current seed position be `p`. The verifier consumes `[seed, d0, ..., d(k-1)]` at positions `p` through `p+k`. It returns continuations `a0` through `ak`.

The core commits `drafts[:j] + [a_j]`. On full acceptance, `j=k`. The target state corresponds to the consumed seed plus the accepted drafts. The continuation is the next seed, not an already consumed target token.

The adapter owns target KV, recurrent state, convolution state, and draft feedback. It must prove their correspondence for every acceptance count. Causal masking can protect unused KV suffixes only with explicit overwrite-before-read evidence.

## Allocation and capture

Do not add another pool without an allocation census. The current feedback pool does not cover lazy draft inputs and every capture scratch allocation.

Prefer one reusable feedback bank with a refreshed prefix-zero entry. Fence its prior use before overwrite. A bounded bank per round remains an alternative only with a correct worst-case memory bound.

The current 64-token, three-draft example allocates 23 banks. Repeated zero acceptance can require 60 speculative rounds before the plain tail. This is a premature-fallback defect, not evidence of an out-of-bounds write.

Trace plans bind physical buffer addresses, shapes, roles, and lifetimes. Package-matched allocator source warns about allocation lifetime overlap. It does not impose a universal ban on every temporary allocation.

A second trace can become resident only after a tested non-overlap contract covers its inputs, outputs, scratch, and first replay. Otherwise, retain a single-trace or trace-release fallback. Capture order alone is not proof of safety.

## Ownership and failure

The pure core owns round, commit, and economic transitions. Move `run_with_watchdog` out of the policy core. Threads, clocks, process exit, and TTNN belong to the shell.

All serving routes submit device work to one owner. The MTP branch currently bypasses the batch worker and eager lock. A request-level lock used only by MTP is insufficient because sampled work shares the same model.

A stalled device thread invalidates normal cleanup. The shell must close admission and end the process without another TTNN call. The supervisor's restart does not authorize a board reset. That effect requires separate operator authority.

Deadline expiry must stop generation, not enter an unbounded plain tail. Plain fallback must first release or safely isolate resident traces. Cleanup covers verifier, draft trace, inputs, feedback, and partially constructed resources.

## Economics

Replace the current boolean-only caller pattern with an explicit next-streak decision. The caller currently increments only after the predicate already requests disengagement, so the streak never leaves zero.

Measure the complete round through rollback and synchronization. Use calibration from the actual fallback route. Keep prefill, trace preparation, complete request time, and steady round time separate.

Fix the undefined `started` variable before a success-path device test. Ruff reports this fault without hardware. A successful pure commit test does not cover the serving return path.

## Validation sequence

First run the existing device-free baseline. Add fake-adapter tests for every acceptance count across multiple rounds, including accept-then-zero-reject and repeated zero rejection.

Add concurrent-route, expired-deadline, partial-allocation, capture-failure, stale-address, and abandoned-worker fixtures. Require no device calls after terminal timeout. Add an ordinary successful return-path test.

Then run a representative simulator state-carry path or preserve its exact unsupported feature. An approved hardware screen starts with the smallest request and expands only after exact state and cleanup pass.

The final screen compares complete greedy token sequences and every relevant state family against the plain control. It includes repeated requests, tail fallback, all rejection positions, and complete economic timing.

## Rollback and non-claims

Keep `QWEN36_MTP_SERVING` off until matched evidence and owner review pass. The diagnostic 1.382x result is not a sustained-serving guarantee. A failed stability gate remains a failure.

This document repair does not change runtime code or mark unfinished tasks complete. Compact selection is a separate follow-on change. Sampled MTP remains out of scope.
