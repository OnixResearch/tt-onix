# Physical result

## Verdict

The complete Python 3.14 roots pass every B=4 admission gate.

- Receipt BLAKE3: `2bf7d74e784d2c5aecfc60a4252f7141955faf1c735d25b94dbc3d832b84b136`
- Measured package envelope: `/nix/store/i086ihcgzxqgdr4y8krzph9mng9xvscq-qwen36-0.1.0`
- Execution identity: `/nix/store/jilqzkn7gp3mxgnh41jgc6326z6n4rnj-qwen36-execution-identity-0.1.0`
- Runtime: `/nix/store/1l9nlvvvzripaa92wbzpglab0rc25ir4-tt-metal-0.74.0`
- Physical devices: `0` and `1`
- Root order: `single,batch,batch,single`

## Typed gates

- Aggregate decode speedup: `3.5475913484735178` (minimum `3.0`)
- Per-user decode retention: `0.8868978371183794` (minimum `0.75`)
- TTFT ratio: `3.1320421887440983` (maximum `4.5`)
- Tail-latency ratio: `1.066116770546012` (maximum `1.5`)
- Token and text parity: pass
- Active prefix and distinct state slices: pass
- Clean reset and physical evidence: pass
- Admission reasons: empty

## Execution identity

The identity artifact binds the Python environment, TT-Metal runtime, model source, service code, patches, decode policy, immutable batch policy, and wrapper contract. It excludes only `batch-admission-binding.ncl`.

The first package design included `default.nix` and changed identity during a receipt-only edit. A second design embedded flake-source paths and also changed identity after unrelated repository edits. Both designs were rejected. The final design uses isolated `builtins.path` inputs, moves wrapper semantics into an immutable Nickel contract, and excludes mutable packaging checks.

## Retry discipline

One earlier partial run was invalidated when a concurrent fleet activation restarted the old production package during `batch-2`. No receipt was produced from that run. The final runner used a temporary runtime mask for the production unit, completed all four roots, removed the mask, and restored production.

## Final deployment

The integrated Onix closure `/nix/store/hd3yw7s6xnxz9nl3cvrg23h5gbrwqdiq-nixos-system-britton-desktop-26.11.20260819.afe3d8a` runs package `/nix/store/nizrk43l4gma8vcyrmvz8v4569s3ka37-qwen36-0.1.0`. The package binds the measured execution identity and receipt with `physically_admitted = true`.

Startup retained `Type=notify` and completed B=1/B=4 warm-up in `104.35764978604857` seconds. The final production smoke measured `16.73992658166973` t/s at B=1 and `57.05838868065472` aggregate t/s at B=4. Normal responses remained admission-ineligible, reset cleanly, and used state slices 0 through 3. The service owned physical devices 0 and 1.

## Evidence locations

- Tracked receipt: `pkgs/qwen36/evidence/2026-08-26/qwen38-27b-batch-admission-receipt.json`
- Raw roots and journals: `.pi/evidence/qwen38/batch-admission-stable-20260826/`
