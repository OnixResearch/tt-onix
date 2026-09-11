## Why

Repeated system prefixes can avoid prefill, but a KV-only cache cannot restore this hybrid model. Each prefix also needs recurrent and convolution state.

Finding 6 in `docs/research/qwen38-p150x2-2026-09-04.md` derives a minimum recurrent snapshot of 72 MiB per chip. A bounded cache needs a measured benefit and a memory budget.

## What Changes

- Add an opt-in cache for complete, immutable hybrid prefixes.
- Bind exact tokens, model content, execution identity, precision, and tenant scope.
- Restore through stable scratch buffers without shared mutable state.
- Compare cold prefill with lookup, restore, eviction, and continuation costs.

## Impact

A pure cache-policy component owns admission, identity, quotas, and eviction decisions. A TTNN adapter owns snapshots and copies. The service shell supplies tenant authority and observed memory facts.

The first candidate covers one approved repeated prefix within the existing context bound. It does not create persistent cross-restart caches or authorize sharing between tenants.
