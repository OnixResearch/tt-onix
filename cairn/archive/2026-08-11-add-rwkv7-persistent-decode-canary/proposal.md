# Change: Add a bounded persistent decode canary

## Why

Persistent decode now emits telemetry and fixed alerts, but critical results only recommend admission disablement. Production still lacks one bounded canary controller that runs matched controls and passes an enforced interlock into every worker.

Without an interlock, a critical monitoring result cannot force later canary workers onto normal dispatch. Without a bounded runner, request count, duration, machine scope, and candidate traffic share remain operator conventions.

## What Changes

- Add a strict Nickel canary policy with fixed request, duration, machine, traffic, profile, window, scheduling, and gain limits.
- Add a pure Rust canary core that creates armed and final interlock receipts.
- Add a Rust canary command that runs a matched physical A/B benchmark under a named timeout.
- Collect only persistent candidate worker telemetry and validate the matched normal controls through the physical A/B receipt.
- Pass a BLAKE3-bound interlock to every worker.
- Force normal dispatch after a held, disabled, malformed, mismatched, or critical interlock.
- Retain a bounded physical canary receipt before production completion.

## Scope

The canary runs on one named host with physical devices `0` and `1`. It does not route live service traffic, mutate the installed Nickel profile, page an external provider, or admit decode window `8`.
