## Why

Python call profiling made eager host-timeline duplicates non-comparable. A low-overhead boundary is required before host enqueue attribution can continue.

## What Changes

- Replace the Python `Profiler` boundary with two Tracy signpost messages around device prefill.
- Export and validate `messages.csv` beside existing C++ host zones.
- Repeat one bounded balanced physical pilot and reject unstable duplicates.

## Impact

- **Files**: Qwen service, host-timeline comparator, evidence, accepted runtime spec.
- **Testing**: focused Qwen and quality checks, comparator self-tests, one P150x2 pilot.
