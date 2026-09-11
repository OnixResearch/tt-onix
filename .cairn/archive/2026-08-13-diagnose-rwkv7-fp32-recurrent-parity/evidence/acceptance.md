# Acceptance

## Verdict

The bounded diagnosis is complete. Window `16` remains blocked.

The first CPU/device recurrent divergence occurs at model step `0`, layer `0`, in `outer_update`. The preceding recurrent operations match exactly.

The device outer product matches a CPU recomputation from the exact BF16-derived device inputs. Therefore, BF16 recurrence-input materialization causes the first mismatch. FP32 recurrence arithmetic does not cause that baseline mismatch.

The bounded correction keeps the embedding output, prior attention vector, and layer-zero weights in FP32. It fails at the same initial boundary. The corrected key and value rows still differ from the pinned CPU authority. The FP32 outer product also differs when the CPU uses the exact corrected device inputs.

The required initial boundary did not match. Therefore, the sixteen-token correction run did not start.

## Decision

- Keep production windows `[2,4,8]`.
- Keep candidates v1 and v2 immutable and rejected.
- Do not create candidate v3.
- Do not start performance, canary, deployment, observation, or admission gates.
- Require a new bounded numerical contract before another correction attempt.

## Evidence

See `bounded-diagnosis.json` for hashes, counts, first mismatch values, maximum errors, correction scope, and non-claims.

All retained evidence is aggregate-only. Temporary raw values, logs, compiler data, caches, and profiler output were removed.
