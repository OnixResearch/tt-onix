## Context

`NativeMtpVerifierAdapter._forward_device` calls `_lm_head`. `replay_round` reads one full vocabulary replica and computes host argmax. The existing plain decode adapter already reduces shard-local winners.

## Decisions

Split diagnostic logits from production selection. The compact verifier keeps logits sharded, computes one winner per continuation row and shard, and merges winners in a pure function. Global token IDs and finite-value checks are explicit inputs.

The sampled candidate uses the same bounded reduction component but a distinct profile. Each shard returns enough local candidates for the existing global top-K. A sub-shard plan keeps local indices within the kernel range. Global offsets use a wide integer type.

The first implementation preserves the current top-K, nucleus, temperature, seeded draw, and greedy first prefill token. It does not claim unrestricted top-p equivalence. New penalties or larger support require a separate coverage proof.

## Boundaries

The core owns winner ordering, tie rules, candidate coverage, and fallback decisions. TTNN owns tile reduction and readback in the adapter. The composition root selects compact or diagnostic output before trace capture. Shape or output-mode changes invalidate the binding.

The existing MTP change must establish safe multi-round state handling before serving integration. Compact selection must not change verifier prefix snapshots or rollback semantics.

## Validation

Positive tests cover every shard and continuation row, all acceptance counts, and identical injected draws. Negative tests cover equal maxima, padding, non-finite values, missing shards, duplicate IDs, index overflow, and unsupported penalties.

A matched physical screen records reduction, readback, merge, full round, and request-wall time. A smaller payload is not acceptance evidence by itself.

## Alternatives and rollback

Do not move arbitrary Python sampling onto the device first. That combines RNG semantics and selection cost in one experiment. Do not import a complete vLLM sampler.

Keep the existing full-logit diagnostic path and current sampled tail as fallbacks. A failed compact candidate remains default-off.
