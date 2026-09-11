## ADDED Requirements

### Requirement: GDN batched prefill supports instance counts above the core count

r[tenstorrent.qwen38_p150x2.gdn_batched_prefill_grid]

The GDN chunk-prefill program factory MUST support a launch whose instance count exceeds the compute-core count. When the instance count is at or below the core count, each core MUST keep exactly one instance and the existing mapping MUST NOT change. When the instance count exceeds the core count, each active core MUST take a contiguous chunk of `ceil(instances / ncores)` instances, and the launch MUST carry the chunk start and chunk length as runtime arguments. The factory MUST reject a malformed or inconsistent instance range before device work. Batched GDN prefill MUST produce exact per-position parity against the serialized per-user prefill reference. The per-user fallback MUST stay reachable and MUST be selected fail-closed for a missing runtime argument or an unsupported shape. This requirement MUST NOT admit any batch width, admit a timing claim, or change any admission binding.

#### Scenario: Batched prefill overflows the core count with exact parity

r[tenstorrent.qwen38_p150x2.gdn_batched_prefill_grid.valid]

- **GIVEN** an instance count above the compute-core count, a valid contiguous chunk assignment, and a warmed model
- **WHEN** the batched GDN chunk prefill runs on physical devices 0 and 1
- **THEN** every core MUST process exactly its assigned contiguous chunk
- **AND** every position MUST match the serialized per-user prefill reference exactly
- **AND** the receipt MUST record the instance count, the core count, and the chunk length.

#### Scenario: The instance count stays within the core count

r[tenstorrent.qwen38_p150x2.gdn_batched_prefill_grid.unchanged]

- **GIVEN** an instance count at or below the compute-core count
- **WHEN** the factory plans the launch
- **THEN** each core MUST receive exactly one instance
- **AND** the produced program mapping MUST equal the pre-change mapping.

#### Scenario: A malformed instance assignment is rejected

r[tenstorrent.qwen38_p150x2.gdn_batched_prefill_grid.invalid]

- **GIVEN** a negative instance count, a zero core count, overlapping chunks, a chunk length below one, a missing runtime argument, or a shape outside the closed set
- **WHEN** the pure instance-assignment core validates the plan
- **THEN** it MUST fail before device work
- **AND** the factory MUST select the per-user fallback
- **AND** the receipt MUST name the bounded fallback reason.
