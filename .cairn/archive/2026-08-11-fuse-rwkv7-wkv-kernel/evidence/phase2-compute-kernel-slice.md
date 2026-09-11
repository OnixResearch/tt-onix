# Fused WKV kernel — first compute-kernel slice

This slice starts the device source for Phase 2. It does not complete K1-K4.

## Source

`pkgs/rwkv7-p150x2/kernel/fused_wkv_compute.cpp` now contains the Blackhole compute kernel for one or more owned work items.

For each `(request, head)` work item, the kernel computes:

- `Sa = S @ a`;
- `S' = S * diag(w) + Sa outer b + v outer k`;
- `o = S' @ r`;
- a BF16 readout in CB 16; and
- an FP32 next-state matrix in CB 17.

The kernel fixes the head width at two 32-element tiles. The state uses four tiles.

## Device-free source guard

`fused_wkv_kernel_contract` validates the source structure. The guard requires every recurrence term, the Blackhole-only branch, separate output boundaries, and balanced input CB waits and pops.

Five tests pass:

- the reviewed source passes;
- a removed decay term fails;
- one head tile fails;
- a missing state pop fails; and
- a Wormhole branch fails.

Clippy passes with warnings denied.

## Package-matched compile gate

`checks.x86_64-linux.rwkv7-p150x2-fused-wkv-architecture` cross-compiles the kernel with the packaged tt-metal 0.74.0 Blackhole TRISC compiler.

The positive source produces a Blackhole object. A mutated Blackhole fixture fails compilation with the required diagnostic.

The compiler reports deprecation warnings for the package-matched `mm_init`, `mm_init_short`, and `transpose_wh` APIs. These warnings are not errors in tt-metal 0.74.0. The later instruction-latency and performance audits must review these calls.

## Non-claims

This result proves source-contract coverage and package-matched compilation only. It does not prove numerical parity, semaphore safety, simulator execution, silicon execution, or performance.

The reader, writer, program factory, runtime fallback shell, and physical admission remain open.
