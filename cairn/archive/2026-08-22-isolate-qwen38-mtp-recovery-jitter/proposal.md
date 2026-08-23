# Isolate Qwen3.8 MTP recovery jitter

## Problem

The device-resident snapshot route preserved exact MTP state and reduced median recovery time by `56.11279598880183%`. Six prompt-route groups still exceeded the `1.25` duplicate-root bound. The current control captures and releases a new MTP trace for every root.

## Proposed change

Reuse one fixed-shape MTP trace and its buffers for all recovery roots in one loaded process. Use the same trace for draft replay, accepted-prefix replay, and committed-token advance. Keep the existing balanced schedule, exact state checks, stability bound, and eager production behavior.

## Success

A matched P150x2 receipt must show one process-level MTP trace capture, exact binding reuse, exact continuation state, and complete cleanup. Every prompt-route recovery spread must be at most `1.25`. The snapshot/re-prefill median ratio must be at most `0.8`.

## Non-claims

This change does not admit MTP, traced target decode, batching, TP4, or production behavior. Device-free checks and simulator results cannot select the mechanism.
