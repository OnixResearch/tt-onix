# Decode chunking compiles and ships

Recorded 2026-09-11 for change `lift-gdn-grid-instance-cap`, tasks I2, I4, I5,
and V4a.

## Result

The tt-metal variant builds. This is compile evidence for the decode chunking,
not a runtime or parity result.

| Item | Value |
| --- | --- |
| Variant derivation | `/nix/store/b5ciph8vx1abipn6179kvf30nn89l217-tt-metal-0.77.0.drv` |
| Variant output | `/nix/store/j9b6liiwxqvfnk4f7dfgz09zf0slvylb-tt-metal-0.77.0` |
| Shared derivation | `/nix/store/ysb1n7p95zzb4klmpzfqnr9dr08c9nnc-tt-metal-0.77.0.drv` |
| Shared output | `/nix/store/dr9bj8qwflylq39zl5ajcx42cdrb1bv9-tt-metal-0.77.0` |

The two derivations and the two outputs are distinct, and the shared output is
still present. The variant is additive.

## Shipped-artifact checks

Checked in the built variant tree:

| Check | Result |
| --- | ---: |
| `gated_delta_decode` binding present in `_ttnn.so` | 5 occurrences |
| `head_count` / `head_start` in `kernels/compute/gated_delta_decode.cpp` | 2 |
| `head_count` / `head_start` in `kernels/dataflow/reader_gated_delta_decode.cpp` | 5 |
| `head_count` / `head_start` in `kernels/dataflow/writer_gated_delta_decode.cpp` | 5 |
| `heads_per_core` in `gated_delta_decode_program_factory.hpp` | 1 |

`heads_per_core` exists only in this change, so its presence in the installed
header proves the factory header patch landed. The kernel counts prove the
instance loops reached the JIT kernel sources that the runtime compiles.

## What this does not show

- The factory and kernel **agreement** is compile-checked only in the sense that
  the kernels compile. The runtime argument counts are matched by inspection,
  not by a run. A mismatch would surface as a runtime error, not a build error.
- No device op has executed. No parity, no occupancy, and no timing result
  exists.
- The prefill SCAN site is not in this variant. `distribute_scan` still carries
  its original guard.
- The package device-free check has not run against the variant. `V5` is open.

## Build note

Two long builds were spent on this. The first was killed to bundle the prefill
mechanism into a single rebuild; the prefill kernels were larger than expected,
so the bundle was abandoned and the decode-only build restarted rather than
risk an unverified kernel change in a multi-hour build.
