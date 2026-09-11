## Why

The current package already uses BF4 MLP gate/up weights. Blanket BF4 is not a new optimization. The retained FP8 MTP-head screen lost all draft acceptance.

Finding 7 in `docs/research/qwen38-p150x2-2026-09-04.md` supports a narrower precision study.

## What Changes

- Evaluate selected MLP down projections before GDN projections or the shared LM head.
- Keep recurrent state FP32 and the MTP head BF16.
- Evaluate BF8 KV as a separate, context-dependent candidate.
- Produce an explicit speed, memory, and quality trade-off table for each recipe.

## Impact

Typed Nickel owns the recipe and thresholds. The pure evaluator owns quality and performance verdicts. Loaders and cache adapters own conversions and distinct cache namespaces.

A prefill score alone cannot admit decode precision. Every candidate needs teacher-forced decode, chunk invariants, and task-quality evidence. Greedy exactness and sampled distributional admission remain separate contracts. No precision default changes in this proposal.
