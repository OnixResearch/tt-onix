# Tasks: Lift the GDN one-instance-per-core grid cap

## Phase 1: Attribution

- [ ] [serial] A1 Confirm both guards in the pinned source: the decode factory guard in `pkgs/tt-metal/gdn-decode-fp32.patch` and `TT_FATAL(BH <= ncores)` in `chunk_gdn_phased_program_factory.cpp`, with their file and line. r[tenstorrent.qwen38_p150x2.gdn_grid_instance_cap]
- [ ] [serial] A2 Record the instance count at TP=2 for B=1, B=4, B=8, and B=16 against the measured compute-core count, and identify the exact batch width where each guard binds. r[tenstorrent.qwen38_p150x2.gdn_grid_instance_cap]
- [ ] [serial] A3 Reproduce the decode guard device-free at B=8 and record the exact diagnostic and any fallback that hides it. r[tenstorrent.qwen38_p150x2.gdn_grid_instance_cap]
- [ ] [serial] A4 Reconcile the prefill guard with the rejected B=8 receipt root cause and record whether one guard can explain both missed bounds. r[tenstorrent.qwen38_p150x2.gdn_grid_instance_cap]

## Phase 2: Implementation

- [ ] [serial] I1 Add the pure instance-assignment core with contiguous chunking and closed range validation. r[tenstorrent.qwen38_p150x2.gdn_grid_instance_cap.invalid]
- [ ] [serial] I2 Lift the guard in the fused GDN decode program factory and pass `(bh_start, n_inst)` as runtime arguments. r[tenstorrent.qwen38_p150x2.gdn_grid_instance_cap]
- [ ] [serial] I3 Lift the guard in the GDN chunk-prefill phased program factory and keep its `BH * cand <= ncores` sub-split search bounded. r[tenstorrent.qwen38_p150x2.gdn_grid_instance_cap]
- [ ] [serial] I4 Keep each per-core instance loop bounded and the copy width dtype-generic, matching the layout fix the decode kernel already carries. r[tenstorrent.qwen38_p150x2.gdn_grid_instance_cap]
- [ ] [serial] I5 Add a tt-metal variant derivation carrying the change; leave the shared `tt-metal` 0.77 output untouched. r[tenstorrent.qwen38_p150x2.gdn_grid_instance_cap]
- [ ] [serial] I6 Bind the variant, the assignment core, and both factories into the package execution identity. r[tenstorrent.qwen38_p150x2.gdn_grid_instance_cap]

## Phase 3: Verification

- [ ] [parallel] V1 Add positive device-free probes for the overflow plan, the chunk boundaries, and the unchanged in-range mapping. r[tenstorrent.qwen38_p150x2.gdn_grid_instance_cap.valid]
- [ ] [parallel] V2 Add negative probes for every rejected plan shape, including a gap, an overlap, and a short interior chunk. r[tenstorrent.qwen38_p150x2.gdn_grid_instance_cap.invalid]
- [ ] [serial] V3 Prove the in-range program mapping is unchanged against the pre-change build for both factories. r[tenstorrent.qwen38_p150x2.gdn_grid_instance_cap.unchanged]
- [ ] [serial] V4 Confirm the full PR applies to the pinned source with zero hunk failures, and record the dry-run result. r[tenstorrent.qwen38_p150x2.gdn_grid_instance_cap]
- [ ] [serial] V5 Run the package build and the device-free check. r[tenstorrent.qwen38_p150x2.gdn_grid_instance_cap]
- [ ] [serial] V6 Run the bounded Blackhole kernel hazard audits over both changed factories and their kernels. r[tenstorrent.qwen38_p150x2.gdn_grid_instance_cap]
- [ ] [serial] V7 Run a fresh Blackhole simulator smoke before hardware. r[tenstorrent.qwen38_p150x2.gdn_grid_instance_cap]
- [ ] [serial] V8 Run matched physical decode at B=8 on devices 0 and 1 and require exact output and recurrent-state parity. r[tenstorrent.qwen38_p150x2.gdn_grid_instance_cap.valid]
- [ ] [serial] V9 Run matched physical prefill and require exact per-position parity against the serialized per-user reference. r[tenstorrent.qwen38_p150x2.gdn_grid_instance_cap.valid]
- [ ] [serial] V10 Confirm the shared `tt-metal` 0.77 output is byte-identical and RWKV-7 and training checks still pass. r[tenstorrent.qwen38_p150x2.gdn_grid_instance_cap.unchanged]

## Phase 4: Completion

- [ ] [serial] D1 Record both mechanism results, the parity tables, the chunk arithmetic, and every non-claim; do not move any admission binding. r[tenstorrent.qwen38_p150x2.gdn_grid_instance_cap]
- [ ] [serial] D2 Run the repository Cairn, artifact, and Qwen package gates. r[tenstorrent.qwen38_p150x2.gdn_grid_instance_cap]

## Verification Coverage

- `Scenario: An instance count above the core count launches with exact parity` -> V1, V4, V5, V6, V7, V8, V9
- `Scenario: The instance count stays within the core count` -> V1, V3, V10
- `Scenario: A malformed instance assignment is rejected` -> I1, V2, V5, D1
