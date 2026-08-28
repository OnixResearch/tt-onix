## Why

Two rejections with receipts (same-width trace reuse; B=8 admission) both attribute their loss to host-side staging: per-user GDN snapshots make an eight-device round trip through the host during prefill, and every decode step recomputes RoPE trig that depends only on position. At B=4 the state-install phase alone costs a median `0.494` seconds per request group.

## What Changes

- Replace the per-user host snapshot round trip in `prefill_paged_slots` with device-resident clones written through the existing `write_slot` path, gated on no parked prefill trace (the upstream clone-aliasing hazard keeps the host path as fallback).
- Defer per-user logits readback to one synchronized batch after the prefill loop.
- Cache per-position RoPE cos/sin tables so each decode step gathers rows instead of recomputing arange/outer/cos/sin.
- Keep the `first_token_callback` evidence boundary (folded into the new patch, superseding `batch-prefill-observation.patch`).
- Add device-free source checks for the staging gate and the RoPE table path.
- Reset the B=4 batch admission binding to not-measured (execution identity changes) and re-run the matched `single,batch,batch,single` admission control on the new artifact before any production claim.

## Impact

- **Files**: `pkgs/qwen36` (new patch, patch list, check.py), the Qwen runtime spec, and physical evidence.
- **Testing**: Package checks, positive and negative profile tests, Cairn gates, deterministic profile export, matched physical B=1 versus B=4 roots, and production restoration.
