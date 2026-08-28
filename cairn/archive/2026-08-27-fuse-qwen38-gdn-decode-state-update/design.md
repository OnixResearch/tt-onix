## Context

The accepted component receipt measured a 55.3-millisecond traced decode step.
Its weighted model assigns 28.5 milliseconds to GDN, 24.0 to MLP, 6.0 to full attention, and 2.1 to the head.

At the pinned dimensions, each device reads about 57.9 MB of GDN projection weights per layer.
At 512 GB/s, this gives a 113-microsecond lower bound.
The measured 594 microseconds leaves about 481 microseconds outside projection weight reads.

## Goal and Success Contract

Reduce the serial GDN decode launch chain without changing model equations, precision, state ownership, or service scheduling.

The change is complete when:

- the candidate remains default-off and fails closed;
- device-free positive and negative checks pass;
- matched P150x2 evidence uses `control,candidate,candidate,control` order;
- generated tokens and text match exactly;
- recurrent and convolution state remain isolated and reset cleanly;
- candidate GDN median time is at most `0.95` of control;
- full decode-step median time is at most `0.97` of control;
- duplicate-root spread is at most `1.25`;
- a selected execution identity receives fresh B=4 admission before deployment.

The bounds are fixed before physical measurement.
They must not move after results exist.

## Portfolio Decision

### Candidate A: full TT-Lang decode kernel

This mechanism removes about 30 launches per GDN layer.
The independent P150 implementation proves the architectural direction.

It is not eligible on the current stack.
The package has no co-versioned `ttl` compiler, and TT-Lang 1.1.6 cannot preserve the required fp32 carry exactly.
Its documented fp32 path truncates through TF32 or produces invalid values through the attempted SFPU route.

### Candidate B: convolution add-chain

This mechanism is already present in the local TP2 implementation.
The state uses separate persistent rows, three copies, one multiply, and three MAC operations.
Repeating it would not change the path.

### Candidate C: projection layout replacement

The prior DRAM-sharded GDN decode candidate was physically slower than the tuned wide 1D path.
The current projection program remains selected.

### Candidate D: package-matched fp32 recurrence kernel

The bounded TTNN-only modes removed up to four launches but missed the fixed full-step gate.
Grouped-head broadcast is unsupported by the 0.74 matmul contract.

The selected full mode keeps fused Q/K RMSNorm and moves the complete fp32 recurrence into one package-matched device program.
The program reads each head state once, writes it in place after use, and returns the updated output.
The `norm` and `state` modes remain isolated controls.

## Architecture

### Functional core

`decode_fusion_candidate.py` owns the closed mode and fail-closed environment parsing.
It imports no TTNN type and performs no I/O except reading the supplied mapping.

### Imperative shell

`gdn/tp.py` allocates trace-stable norm constants during layer construction.
It passes the selected mode into the recurrent decode adapter.

`ttnn_delta_rule_ops.py` owns TTNN orchestration and keeps the baseline branch unchanged.
The package-matched runtime owns program construction, CB flow, fp32 compute, and trace-safe state output.

## Numerical Contract

The candidate keeps fp32 q, k, v, beta, gate, and recurrent-state arithmetic.
It changes kernel grouping, so physical validation must check exact generated output and bounded state equivalence.
No PCC-only result can authorize production output changes.

## Validation Strategy

1. Run the existing package check before implementation.
2. Run the package build and device-free checks after implementation.
3. Measure `off`, `norm`, `state`, and `full` as a bounded candidate set.
4. Reject modes with errors, token drift, state drift, reset failure, or insufficient speed.
5. Run full B=4 admission only for the strongest valid mode.
6. Keep `off` selected if no mode passes every fixed gate.

## Risks and Mitigations

- **Rounding changes:** require exact token and text parity plus state checks.
- **Trace capture allocation:** allocate norm constants before any trace.
- **Unknown gate values:** map them to `off`.
- **Partial norm inputs:** reject them before device execution.
- **False local win:** require full-step and B=4 evidence after component timing.
- **Execution drift:** bind the patch to the Nix execution identity.

## Non-Goals

- Proving that recurrence fusion reaches the 113-microsecond floor.
- Claiming that the unavailable TT-Lang kernel is accepted.
- Changing any admission bound after measurement.
