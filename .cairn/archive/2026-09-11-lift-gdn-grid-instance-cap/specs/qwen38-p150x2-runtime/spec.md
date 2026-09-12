## ADDED Requirements

### Requirement: GDN decode and prefill kernels support instances above the core count

r[tenstorrent.qwen38_p150x2.gdn_grid_instance_cap]

The fused GDN decode program factory and the GDN chunk-prefill phased program factory MUST support a launch whose program-instance count exceeds the compute-core count. When the instance count is at or below the core count, each core MUST keep exactly one instance and the produced program mapping MUST equal the pre-change mapping. When the instance count exceeds the core count, each active core MUST take a contiguous chunk of `ceil(instances / ncores)` instances, and the launch MUST carry the chunk start and chunk length as runtime arguments. The factory MUST finish the assignment with a single pass over the active cores, so the chunk start MUST equal `core_index * chunk_length` and the chunks MUST partition `[0, instances)` with no gap, overlap, or short interior chunk. The repository MUST own the chunk arithmetic and its range validation in a pure, device-free core that rejects a zero or negative core count, a zero or negative instance count, a chunk length below one, and any assignment that does not partition the instance range. The decode path MUST produce exact output and recurrent-state parity against the admitted fused reference. The prefill path MUST produce exact per-position parity against the serialized per-user reference. This requirement MUST NOT admit any batch width, admit a timing claim, or change any admission binding.

#### Scenario: An instance count above the core count launches with exact parity

r[tenstorrent.qwen38_p150x2.gdn_grid_instance_cap.valid]

- **GIVEN** an instance count above the compute-core count, a valid contiguous chunk assignment, and a warmed model
- **WHEN** the GDN decode kernel or the GDN chunk prefill runs on physical devices 0 and 1
- **THEN** every core MUST process exactly its assigned contiguous chunk and no core MAY process an overlapping chunk
- **AND** decode output and recurrent state MUST match the admitted fused reference exactly
- **AND** prefill MUST match the serialized per-user reference at every position
- **AND** the receipt MUST record the instance count, the core count, and the chunk length.

#### Scenario: The instance count stays within the core count

r[tenstorrent.qwen38_p150x2.gdn_grid_instance_cap.unchanged]

- **GIVEN** an instance count at or below the compute-core count
- **WHEN** either factory plans its launch
- **THEN** each core MUST receive exactly one instance
- **AND** the produced program mapping MUST equal the pre-change mapping
- **AND** every existing decode and prefill receipt for that mapping MUST remain valid.

#### Scenario: A malformed instance assignment is rejected

r[tenstorrent.qwen38_p150x2.gdn_grid_instance_cap.invalid]

- **GIVEN** a zero or negative core count, a zero or negative instance count, a chunk length below one, or a chunk set that leaves a gap, overlaps, or truncates a short interior chunk
- **WHEN** the pure instance-assignment core validates the plan
- **THEN** it MUST fail before any device work
- **AND** the receipt MUST name the bounded rejection reason
- **AND** no factory MAY fall back to a partially valid assignment.
