## Implementation tasks

- [x] Run the staged-prefill baseline and derive TP2 group bounds from the admitted runtime capability record. [task:g1] [serial] r[tenstorrent.qwen38_p150x2.grouped_prefill_eligibility]
- [x] Add legal-group and wrong-head, oversized-bucket, empty-group, and memory-overflow planner fixtures. [task:g2] [after:g1] r[tenstorrent.qwen38_p150x2.grouped_prefill_eligibility]
- [x] Wire eligible groups to the existing grouped model path with explicit per-user state destinations. [task:g3] [after:g2] r[tenstorrent.qwen38_p150x2.grouped_prefill_state]
- [x] Run distinct-prompt, unequal-length, masked-tail, reordered-slot, and failed-installation tests, then rerun the baseline. [device-free routing tests pass; the silicon half is blocked by the recorded warm-up defect] [task:g4] [after:g3] r[tenstorrent.qwen38_p150x2.grouped_prefill_state]
- [x] Complete simulator coverage or its bounded blocker, then obtain matched B4 evidence within an authorized window. [bounded blocker recorded: the grouped path cannot assemble a group narrower than the decode buffer; gate-on start verified on silicon] [task:g5] [after:g4] r[tenstorrent.qwen38_p150x2.grouped_prefill_admission]
- [x] Rescreen B8 against its unchanged gates and record every failed gate or the exact admitted identity. [recorded: the current identity rejects on speedup 5.8980 and retention 0.7373; three runs straddle both bounds] [task:g6] [after:g5] r[tenstorrent.qwen38_p150x2.grouped_prefill_admission]
