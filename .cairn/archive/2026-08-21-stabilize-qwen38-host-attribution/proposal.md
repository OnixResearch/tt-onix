## Why

Single-request Tracy roots remained unstable after Python profiling was removed. Host attribution needs a bounded sampling method that distinguishes request noise from repeatable mode differences.

## What Changes

- Measure five signposted requests per immutable root and aggregate each root by median.
- Pin the diagnostic service to a fixed CPU set and keep capture attachment outside measured intervals.
- Repeat the exact balanced physical order and fail closed on the existing `1.25` duplicate bound.

## Impact

The comparator and physical runner change. Production admission and eager defaults do not change.
