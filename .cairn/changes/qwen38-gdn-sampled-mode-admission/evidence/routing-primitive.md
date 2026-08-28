# Routing primitive evidence (2026-08-28)

The dual-trace probe on package `hn61wf5g...` (identity `yv113vpa...`)
settled the D1 routing design by measurement.

## Dual residency: falsified

- The off-mode trace captured first replays exactly (`0.0` diff) and stays
  exact through interleaved replays.
- The full-mode trace captured while the off trace is resident replays
  garbage (`2.998e+20` diff): its output buffers are corrupted.
- This reproduces, for decode traces on runtime 0.74, the same corruption
  class the Adartras bundle documented for verify traces.

## Release-then-recapture switching: proven

Four captures around the cycle full->off->off->full, each with nothing else
resident, all replay exactly (`0.0` diff in every direction). Capture cost
0.39-0.75 seconds per switch, amortized over a 64-token request.

Probe: `.pi/evidence/qwen38/gdn-sampled-dist-screen-20260827/dual-trace-probe.json`
(copied to `s3a-screen/`).

## Implemented primitive

- `qwen36_service/sampled_mode_router_core.py` (pure): request temperature
  to route classification (missing and zero -> deterministic, positive
  finite -> sampled, everything else fails closed), switch decision, and
  receipt-mode to route mapping with the sampled admission gate. An empty
  or non-sampled admission permits only the deterministic route.
- `TracedDecodeAdapter.switch_fusion_mode(mode)` releases the resident
  decode trace, then flips `_decode_fusion_mode` on every TP GDN layer,
  which the decode forward reads at call time. The next `ensure_trace`
  recaptures in the new mode. `fusion_mode` reports the resident mode and
  fails closed if layers disagree.
- Device-free probes in `check.py` cover classification positives, invalid
  temperatures, switch decisions, admission gating, the stub-adapter
  release-before-flip order, and unknown-mode rejection.

One bug the probe caught before it could ship: the receipt names the
serving mode `sampled` while the adapter flips between `off` and `full`;
the admission gate now accepts both vocabularies explicitly.

## Remaining D1 work

- Thread the request temperature through `server.py` `generate()`, route
  per request, and call `switch_fusion_mode` on mode changes.
- Physical router validation: greedy request after sampled traffic must
  reproduce the exact-parity tokens of the off path; a sampled request must
  run the fused path at the measured step time.
- Deploy the changed identity as a new service generation.
