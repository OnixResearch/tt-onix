## Why

The adopted model already uses tuned 1D matmul grids. Some grid choices came from TP4 shapes, while the local TP2 projections have different dimensions.

The wrapper named `tt_all_reduce` performs reduce-scatter on a line mesh. Optimization must use the lowered operation and actual payload, not its name.

## What Changes

- Record an exact TP2 projection and collective inventory.
- Measure collective startup, tile padding, reshape costs, and full-step contribution.
- Evaluate a bounded shape-grid or small-message collective candidate only after attribution.
- Keep TP2 placement and numerical semantics unchanged.

## Impact

The Qwen core owns shape eligibility and candidate selection. TTNN adapters own grids, collectives, and profiling. Existing component probes remain the starting implementation.

Finding 8 in the September research report records source evidence and rejected alternatives. No new topology, firmware setting, network link count, or general sharding framework becomes a default through this proposal.
