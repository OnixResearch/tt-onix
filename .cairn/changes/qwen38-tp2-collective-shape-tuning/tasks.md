## Implementation tasks

- [ ] Run the component baseline and record exact TP2 shapes, layouts, dtypes, grids, and lowered collective sites. [task:t1] [serial] r[tenstorrent.qwen38_p150x2.tp2_operation_inventory]  (baseline and collective site done: 112 collectives per step at 25.1-30.0 us, about 3 ms total; component and traced tables in evidence/step-attribution-2026-10-21.md. The exact projection grids and layouts remain to be inventoried.)
- [ ] Add source-to-trace inventory validation with wrong-mesh, wrapper-name, padding, and missing-link negative fixtures. [task:t2] [after:t1] r[tenstorrent.qwen38_p150x2.tp2_operation_inventory]
- [ ] Select one measured shape or collective mechanism and define its bounded candidate profile. [task:t3] [after:t2] r[tenstorrent.qwen38_p150x2.tp2_shape_candidate]
- [ ] Implement the narrow adapter change and run positive parity, invalid-grid, synchronization, and applicable hazard checks. [task:t4] [after:t3] r[tenstorrent.qwen38_p150x2.tp2_shape_candidate]
- [ ] Rerun the baseline and representative simulator path, or preserve its exact unsupported-path blocker. [task:t5] [after:t4] r[tenstorrent.qwen38_p150x2.tp2_tuning_admission]
- [ ] Compare matched component, full-step, and request timings within an approved hardware window and record the final verdict. [task:t6] [after:t5] r[tenstorrent.qwen38_p150x2.tp2_tuning_admission]
