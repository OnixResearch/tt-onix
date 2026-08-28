## Why

Traced decode is `6.942337116061393` times faster than eager decode, but its matched TTFT ratio is `1.1069467490614955`. Admission requires a ratio no more than `1.05`. Decode trace capture starts after the first token, so capture time cannot explain this TTFT result. A same-process control must separate cache-capacity cost, parked-trace cost, and unmatched process drift before another optimization is selected.

## What Changes

- Add a diagnostic-only TTFT control with a typed Nickel profile.
- Compare request-sized and trace-capacity KV caches with one loaded model and one bounded page table.
- Compare the trace-capacity prefill path before, during, and after a parked decode trace.
- Use a deterministic core to reject wrong route order, identity drift, token mismatch, unstable roots, invalid widths, and malformed durations.
- Preserve eager production and select no optimization when the evidence does not identify a stable mechanism.

## Impact

- **Files**: Qwen TTFT control core, TTNN adapter, command wrapper, typed profile, tests, physical evidence, and adoption documentation
- **Testing**: positive and negative core tests, package gates, focused repository-quality gates, and at most two bounded physical P150x2 outages
