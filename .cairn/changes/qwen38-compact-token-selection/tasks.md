## Implementation tasks

- [ ] Run the selection baseline and define a compact output contract with explicit global IDs and tie rules. [task:c1] [serial] r[tenstorrent.qwen38_p150x2.compact_greedy_selection]
- [ ] Implement the shard-local verifier tail with full-logit parity, padding, tie, non-finite, and missing-shard fixtures. [task:c2] [after:c1] r[tenstorrent.qwen38_p150x2.compact_greedy_selection]
- [ ] Define a bounded sub-shard sampled plan and prove coverage for the existing top-K transforms. [task:c3] [after:c2] r[tenstorrent.qwen38_p150x2.compact_sampled_selection]
- [ ] Add identical-draw, index-boundary, duplicate-ID, unsupported-penalty, and insufficient-support tests, then rerun the baseline. [task:c4] [after:c3] r[tenstorrent.qwen38_p150x2.compact_sampled_selection]
- [ ] Run representative simulator selection or retain its exact unsupported-path blocker before approved hardware use. [task:c5] [after:c4] r[tenstorrent.qwen38_p150x2.compact_selection_admission]
- [ ] Compare complete greedy and sampled costs separately, and record admission or rejection without changing defaults automatically. [task:c6] [after:c5] r[tenstorrent.qwen38_p150x2.compact_selection_admission]
