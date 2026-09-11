## Implementation tasks

- [ ] Run the precision baseline and define typed per-layer recipes with FP32 recurrence and BF16 MTP head fixed. [task:f1] [serial] r[tenstorrent.qwen38_p150x2.precision_recipe_identity]
- [ ] Implement distinct cache namespaces with valid-recipe, stale-cache, scale, dtype, and missing-layer rejection fixtures. [task:f2] [after:f1] r[tenstorrent.qwen38_p150x2.precision_recipe_identity]
- [ ] Add teacher-forced decode, chunk-invariant, nonzero-state, finite-value, and task-quality gates before hardware measurement. [task:f3] [after:f2] r[tenstorrent.qwen38_p150x2.precision_quality]
- [ ] Run positive and deliberate-drift fixtures, then rerun the package baseline and representative simulator path. [task:f4] [after:f3] r[tenstorrent.qwen38_p150x2.precision_quality]
- [ ] Screen MLP down recipes and BF8 KV separately within an approved window with actual memory and complete latency measurements. [task:f5] [after:f4] r[tenstorrent.qwen38_p150x2.precision_frontier]
- [ ] Publish every recipe's speed-memory-quality result and retain current defaults unless the owner approves an admitted recipe. [task:f6] [after:f5] r[tenstorrent.qwen38_p150x2.precision_frontier]
