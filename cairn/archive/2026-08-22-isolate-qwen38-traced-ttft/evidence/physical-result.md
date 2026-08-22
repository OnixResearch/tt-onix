# Physical result

## Scope

The final control used package `/nix/store/ldrskhvsbvd0chs1bs4x4hgb3wsisgz7-qwen36-0.1.0` and TT-Metal `/nix/store/mg0ha5nsvnir1y1ysiyyzmm4hk83z3m6-tt-metal-0.74.0`. It opened physical PCIe device IDs `0` and `1` as a `1x2` mesh. The process used CPUs `8-31`.

The command used one model, stable request-sized and trace-capacity cache sets, a two-block page table, one warm-up per root, and five retained samples per root. Every retained sample returned first token `11751`.

## Pilot

The pilot stopped before mesh open because the command read `DecodeProfile.model_source_revision`. The profile exposes that value as `runtime_source_revision`. The exact error was:

```text
AttributeError: 'DecodeProfile' object has no attribute 'model_source_revision'. Did you mean: 'runtime_source_revision'?
```

The bounded correction changed the field access and added a positive and negative profile-binding test. The eager service returned to healthy state after the pilot.

## Final control

The final route order was:

1. request-sized cache without a parked trace;
2. trace-capacity cache without a parked trace;
3. trace-capacity cache with a parked trace;
4. trace-capacity cache with a parked trace;
5. trace-capacity cache after trace release; and
6. request-sized cache without a parked trace.

The observed causal ratios were:

| Comparison | Prefill-device ratio | Readback ratio | TTFT ratio |
| --- | ---: | ---: | ---: |
| Trace-capacity cache / request cache | `1.0078726281898032` | `0.9332254347335398` | `1.0062416611969702` |
| Parked trace / untraced capacity cache | `0.9958567860794341` | `1.041555784380982` | `0.9951621585112468` |
| Parked capacity route / request route | `1.0036967962865297` | `0.9720063496781749` | `1.0013736235407196` |

All TTFT ratios stayed below the material threshold of `1.05`. Every within-root TTFT spread and duplicate-root TTFT ratio stayed within `1.25`.

The strict evaluator still rejected the control. The first-token readback spread exceeded `1.25` in these roots:

- root `3`: `1.424056552628308`;
- root `4`: `1.3227102062387852`; and
- root `6`: `1.517837724791448`.

The retained readback medians were between `0.011569678987143561` and `0.013851841009454802` seconds. These small intervals were not stable enough for the typed all-phase gate.

TT-Metal warned that buffer allocation occurred while the decode trace was active. The control never executed that trace. This warning prevents a post-prefill trace-correctness claim from this receipt.

## Decision

The final receipt is invalid for causal selection. It selects neither cache capacity nor parked-trace state. The observed same-process TTFT ratios do not reproduce the prior `1.1069467490614955` cross-process ratio, but rejected evidence cannot prove process drift.

The two-outage budget is exhausted. No kernel, cache, trace, precision, collective, or production change is selected. Production returned to the eager service and passed health on port `8000`.

## Evidence

- receipt: `pkgs/qwen36/evidence/2026-08-22/qwen38-27b-traced-ttft-control-receipt.json`
- receipt BLAKE3: `fd127f0cf88d8393487dc95949b072f334037ab2cf3459db914db7e93660f759`
- package: `/nix/store/ldrskhvsbvd0chs1bs4x4hgb3wsisgz7-qwen36-0.1.0`
- runtime: `/nix/store/mg0ha5nsvnir1y1ysiyyzmm4hk83z3m6-tt-metal-0.74.0`

## Post-run hardening

The adversarial device-free pass added an exact five-sample minimum, closed profile keys, and partial cache-allocation cleanup. These changes affect invalid configuration and allocation-failure paths only. The hardened package is `/nix/store/1bl5idd70wrqhxpfajdjrymdhlhapsv8-qwen36-0.1.0`.

The physical receipt remains bound to `/nix/store/ldrskhvsbvd0chs1bs4x4hgb3wsisgz7-qwen36-0.1.0`. The outage budget was exhausted, so the hardened package has no new physical claim.

## Validation

The Qwen package, authority-negative, NixOS service, Python, Nix, Nickel, Cairn, artifact, and repository-quality gates passed. Ruff, Python compilation, Nix formatting, and Cairn lifecycle gates also passed.

Global Tracey coverage retains the existing repository-wide scope gap. Its evidence roots scan only `crates` and `tools`, so it lists this package-owned Python requirement with the other missing package requirements. Focused package and Cairn gates passed.

## Non-claims

- The invalid control does not prove that process drift caused the earlier TTFT result.
- The control does not admit traced decode.
- The control does not prove that a parked trace remains correct after prefill allocations.
- The result does not apply to TP4, batching, concurrency, long context, or thermal behavior.
