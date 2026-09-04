## ADDED Requirements

### Requirement: Exact TP2 operation inventory

r[tenstorrent.qwen38_p150x2.tp2_operation_inventory]

The inventory MUST bind projection shapes, dtypes, tile padding, worker grids, output layouts, and lowered collectives to the execution identity. It MUST distinguish wrapper names from actual reduce-scatter or all-gather operations. Physical link and timing claims MUST come from observed hardware evidence.

#### Scenario: Exact line-mesh inventory
- GIVEN the current TP2 source and a matched operation trace
- WHEN the analyzer records collective sites
- THEN it identifies actual operations and separates semantic sites from kernel launches.

#### Scenario: TP4 or wrapper-only estimate
- GIVEN TP4 dimensions or only the name `tt_all_reduce`
- WHEN the analyzer builds a TP2 cost claim
- THEN it rejects the unsupported attribution.

### Requirement: Bounded shape or collective candidate

r[tenstorrent.qwen38_p150x2.tp2_shape_candidate]

The pure planner MUST select only predeclared shapes, grids, and collective capabilities. The adapter MUST preserve reduction dtype, dependency order, state layout, and mesh identity. Each candidate MUST change one measured mechanism. Unsupported shapes MUST keep the existing tuned path.

#### Scenario: Eligible TP2 projection
- GIVEN a measured projection and an eligible candidate grid
- WHEN the adapter runs the candidate
- THEN output and carried state satisfy the reference contract without extra unbounded reshards.

#### Scenario: Invalid grid or synchronization
- GIVEN an unsupported worker grid, missing barrier, or incompatible collective
- WHEN preflight or the hazard audit runs
- THEN it rejects the candidate before hardware admission.

### Requirement: Full-step tuning admission

r[tenstorrent.qwen38_p150x2.tp2_tuning_admission]

The evaluator MUST compare exact TP2 component and full-step timings under balanced matched roots. It MUST include reshape, synchronization, and request latency costs. Kernel changes MUST pass applicable LLK and NoC hazard audits. A component-only gain MUST NOT establish service improvement.

#### Scenario: Composed improvement
- GIVEN a candidate with passing numerical and hazard evidence
- WHEN component, full-step, and service gates all pass
- THEN the receipt admits only the tested shape and execution identity.

#### Scenario: Isolated benchmark win
- GIVEN a faster isolated matmul but slower full decode
- WHEN the evaluator applies the full-step policy
- THEN it rejects the candidate and preserves the tuned control.
