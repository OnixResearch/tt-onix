# Both chunking sites compile and ship

Recorded 2026-09-11 for change `lift-gdn-grid-instance-cap`, tasks V4a and V4b.

This supersedes `decode-variant-build.md`, which covers the decode site alone.

## Result

The variant builds with both chunking patches. This is compile and packaging
evidence. It is not a runtime, parity, occupancy, or throughput result.

| Item | Value |
| --- | --- |
| Variant derivation | `/nix/store/3k6yjsvdy005s8vlzkl7hkkld48pd92n-tt-metal-0.77.0.drv` |
| Variant output | `/nix/store/6ccynlvv46p54pmp2lb7dbs65hkyj5w9-tt-metal-0.77.0` |
| Shared derivation | `/nix/store/ysb1n7p95zzb4klmpzfqnr9dr08c9nnc-tt-metal-0.77.0.drv` |
| Shared output | `/nix/store/dr9bj8qwflylq39zl5ajcx42cdrb1bv9-tt-metal-0.77.0` |
| Build exit | `0` |

The shared output is still present. The variant is additive.

## Shipped prefill kernels

| Kernel | `wi_count` | instance loop |
| --- | ---: | ---: |
| `kernels/dataflow/reader_chunk_gdn_scan.cpp` | 3 | 1 |
| `kernels/dataflow/writer_chunk_gdn_scan.cpp` | 3 | 1 |
| `kernels/compute/chunk_gdn_scan.cpp` | 3 | 1 |

## Shipped decode kernels

| Kernel | `head_count` |
| --- | ---: |
| `kernels/dataflow/reader_gated_delta_decode.cpp` | 3 |
| `kernels/dataflow/writer_gated_delta_decode.cpp` | 3 |
| `kernels/compute/gated_delta_decode.cpp` | 2 |

`gated_delta_decode` appears 5 times in the variant's `_ttnn.so`.

## Why the guard removal is proven indirectly

The prefill guard lives in `chunk_gdn_phased_program_factory.cpp`, which is not
installed: the runtime ships headers and JIT kernel sources, not every C++
source. The guard's removal is therefore established by three facts together:

1. the patch removes it, which a grep of the patched tree confirms at zero
   occurrences of `TT_FATAL(BH <= ncores`;
2. the patch applies to pristine source under `-p1 --fuzz=0` with no reject;
3. the derivation carrying that patch builds.

The three shipped scan kernels carrying the instance loop is the direct
artifact-level check, because the runtime compiles those sources at run time.

## What this does not show

- No device op has executed. No parity, occupancy, or timing result exists.
- Runtime argument counts between the factories and the kernels agree by
  inspection, not by a run. A mismatch would surface at run time, not at build
  time. This is the main open risk in both sites.
- The package device-free check has not run against the variant. Task `V5` is
  open.
- Neither change is admitted. No admission binding moved.

## Build cost

Four build attempts were spent: one killed to attempt a bundle, one decode-only
that succeeded, and this combined one. The prefill site alone was not built
separately.
