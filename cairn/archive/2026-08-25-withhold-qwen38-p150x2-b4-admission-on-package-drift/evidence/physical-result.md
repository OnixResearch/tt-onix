# Physical result

## Verdict

Formal B=4 production admission is withheld because the measured and deployed package identities differ.

The matched candidate receipt remains valid for `/nix/store/ibjk9b3lynppx1swbvi6x9qg07pdkpp8-qwen36-0.1.0`. The deployed Onix package is `/nix/store/by2xl98jh01cjakf16zck4py9smbhmsr-qwen36-0.1.0` and uses Python 3.14. The profile returns to `physically_admitted = false`.

## Candidate evidence retained

- Receipt BLAKE3: `176d912457f4c0048be90c9b477709e4360dae0681212cf275d108f3ca48465f`
- Aggregate speedup: `3.5516293837023656`
- Per-user retention: `0.8879073459255914`
- TTFT ratio: `3.4362760366231644`
- Tail-latency ratio: `1.220338917916907`
- Token, text, state-slice, active-prefix, and reset checks: pass

These values describe the candidate package only.

## Deployed smoke

Closure `/nix/store/n6d1k61cjqggqv67p02x5an3dpscxsx8-nixos-system-britton-desktop-26.11.20260819.afe3d8a` passed readiness and retained `Type=notify`.

- B=1: 16.4483 t/s, internal TTFT 5.0714 seconds, clean reset
- B=4: 55.6201 aggregate t/s, 13.9050 per-user t/s, internal TTFT 14.8041 seconds, clean reset
- B=4 state slices: complete prefix 0 through 3
- Physical device ownership: devices 0 and 1
- Normal production evidence remained admission-ineligible

This single deployed smoke is not the required two-root matched admission run.

## Next admissible path

A later change must split immutable execution payload identity from mutable admission metadata, or rerun complete matched roots on a stable deployed artifact. It must not transfer the Python 3.13 receipt to the Python 3.14 package.
