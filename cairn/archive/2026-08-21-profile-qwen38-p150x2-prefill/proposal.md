## Why

The traced-decode candidate reduced decode time but failed the physical TTFT gate. Its median device-prefill time remained about `0.114` seconds slower than eager prefill. Aggregate phase timing does not identify the responsible device program.

A new optimization would be speculative without operation-attributed physical evidence. The repository needs a bounded diagnostic that compares eager and candidate prefill programs on the same P150x2 workload.

## What Changes

- Add an opt-in prefill diagnostic that collects package-matched TT-Metal real-time program records.
- Convert vendor records into a deterministic, validated program-signature summary.
- Mark profiled responses as diagnostic and ineligible for TTFT admission.
- Add a deterministic comparator for balanced eager and candidate physical roots.
- Select an optimization only when matched physical evidence attributes a material difference.

## Impact

- **Files**: Qwen service profiling core and adapter, package checks, a profile receipt script, Qwen runtime specification, and physical evidence.
- **Operations**: Production remains eager and does not enable profiling. Physical profiling needs one bounded service outage on PCIe devices `0` and `1`.
- **Testing**: Positive and negative core tests, adapter boundary tests, receipt self-tests, focused Qwen Nix checks, and matched physical profiling.
