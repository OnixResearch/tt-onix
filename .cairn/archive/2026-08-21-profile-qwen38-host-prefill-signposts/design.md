## Context

The package-matched Tracy tools expose C++ enqueue zones, but Python call profiling perturbed eager roots by up to `9.21x`.

## Decision

Use `tracy.signpost` only. Emit `QWEN_PREFILL_START` immediately before device prefill and `QWEN_PREFILL_END` after synchronization. Do not enable `sys.setprofile`. The comparator reads message timestamps, filters existing C++ zones to that interval, and keeps the `1.25` duplicate-ratio gate.

## Non-Goals

No admission change, optimization selection without stable duplicates, runtime uplift, or device-profiler use.
