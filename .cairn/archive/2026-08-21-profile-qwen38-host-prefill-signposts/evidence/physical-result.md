# Physical result

The exact P150x2 order was `eager,candidate,candidate,eager`. Each root used one warm-up outside capture and one measured request. All responses were diagnostic and ineligible for admission. Production was restored healthy and eager.

Package-matched Tracy signposts bounded prefill without Python call profiling. All captures contained ordered start/end messages and C++ enqueue zones.

Duplicate stability still failed the `1.25` bound:

- eager: wall `2.0190806531120726`, enqueue `3.174942201847243`, gaps `1.7997211829801623`;
- candidate: wall `1.3129653451778505`, enqueue `1.4077723936685347`, gaps `1.2972753465032332`.

Verdict: `rejected`, reason: `duplicate-root-instability`. No attribution or optimization is selected. The remaining blocker is host-run variability outside Python profiling. A future attempt must control capture attachment and CPU scheduling, or use an external low-overhead probe with stable duplicates.

Receipt: `pkgs/qwen36/evidence/2026-08-21/qwen38-27b-host-signpost-attribution-receipt.json`

BLAKE3: `3ab6c5ceeab4160699bf68b55ff7407d625b9b9ffd56aa39b076be80b50a48ac`
