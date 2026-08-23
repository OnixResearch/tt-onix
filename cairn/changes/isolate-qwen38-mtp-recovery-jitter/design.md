# Design

## Functional core

The MTP rollback core owns the version-two profile, exact route schedule, trace-capture count, binding rules, state parity, recovery stability, and selection decision. It receives immutable observations and performs no I/O.

## Imperative shell

The MTP adapter owns one fixed-shape trace for the complete loaded process. The shell captures the trace during discarded warmup, reuses its stable inputs and outputs, and closes it before device teardown.

The recovery adapter restores rejected rows first. It then replays accepted drafts that exceed the two-row feedback snapshot. Finally, it advances one committed target token through the same persistent trace.

The re-prefill route uses the same persistent trace after MTP prefill. This control removes repeated eager decode allocation from both routes without changing route semantics.

## Evidence boundary

The receipt records one process-level trace capture, one stable binding, route samples, exact next-round state, and cleanup. Selection requires six roots per route and prompt, exact parity, spread at most `1.25`, and ratio at most `0.8`.

Trace capture stays outside replay economics because warmup performs the only capture. A capture count other than one rejects the result.

## Failure handling

Any trace allocation, capture, replay, binding, state, or cleanup error rejects the result. The shell releases partial trace state on every error path. Production wrappers remain unchanged.
