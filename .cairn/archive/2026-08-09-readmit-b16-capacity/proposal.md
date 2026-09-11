## Why

The archived `scale-rwkv7-card-local-batching` change kept B4 as the admitted
capacity and retained B16 as a measured blocker. It rejected B16 because its
per-request p99 wall latency regressed against B4 under a 0.10 rule.

A stalled-wave investigation on physical cards showed that rule compared the
wrong metric. Per-request wall latency grows with batch concurrency, so it is
not comparable across different capacity workloads. The honest cross-capacity
latency metric is time per generated token. Under that metric, B16 is faster
than B4 in both aggregate throughput and per-token latency, with exact token
parity in every run. B16 should be re-admitted.

## What Changes

- The capacity admission gate compares p99 time per generated token, not
  per-request wall latency.
- The evidence package retains order-balanced B4/B16 physical runs that
  establish B16 as the admitted capacity.
- Documentation and the runtime README state the re-admission and the
  residual 16-wide kernel-stall follow-up.
- The runtime requires a `capacity` object in every fast-worker request, so
  the launcher selects B16 explicitly; the runtime stays fail-closed for an
  unsupported capacity.

## Impact

- **Runtime**: `pkgs/rwkv7-p150x2` accepts B16 as the card-local capacity.
- **Evidence**: `pkgs/rwkv7-p150x2-evidence` ships
  `physical-capacity-b16-admitted/` retention.
- **Gate**: `capacity-evaluate` uses per-token p99.
- **Testing**: per-token p99 positive and negative cases; retention check
  binds the re-admission decision.
