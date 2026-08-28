# D1d router validation: partial pass, blocked on an owner decision (2026-08-28)

Validation window: package `wjkngifv...` (identity `fdkhzgd0...`), served on
a private port with `QWEN36_SAMPLED_FUSION=1`, production stopped and
restored.

## What passed

- The router wiring is live on real traffic. The first greedy request
  served with `serving_route: "off"`, `fusion_mode_switched: false`, tokens
  `[11751, 13, 198, 760]` (the known off-path sequence), 16.91 tok/s
  through the batch traced path.

## The wall

The sampled request never reaches the router. `_validate_common_fields`
(`core.py`, "only greedy temperature 0 is supported") rejects every
`temperature != 0` request with a 400 before routing runs. The serialized
service is greedy-only by long-standing policy; sampling happens on the
host side after generation or not at all.

Combined with the sampled-only scoping of the S3 admission, this means the
fused kernel cannot serve any traffic this endpoint actually accepts. The
router, the switch primitive, and the receipts are all ready; there is no
eligible traffic.

## The owner decision this forces

1. **Implement sampling in the service** for `temperature > 0` (per-step
   distribution readback plus host sampling; the MTP selection machinery
   already owns the candidate-buffer pattern). Largest work; the fused path
   then serves genuinely sampled traffic as scoped.
2. **Extend the admission to greedy traffic**: serve `temperature = 0`
   through the fused kernel under the same distributional gate. The
   measured evidence says greedy divergences are rare near-tie flips (3 of
   4 archived 64-token prompts, one token each; stable agreement 1.0 over
   220 positions), but this drops the exact-parity guarantee the accepted
   spec currently gives deterministic requests. Requires a spec change and
   a new validated policy revision.
3. **Park the enablement**: the router ships inert behind the default-off
   gate, behavior-identical to today, until option 1 or 2 is chosen.

Evidence: `.pi/evidence/qwen38/router-validation-20260828/` (greedy-1.json,
empty sampled.json, server log with the 400).
