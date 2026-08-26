# Search frame

## Goal

Raise admitted Qwen P150x2 served throughput by formalizing the width the deployed coordinator already serves without evidence.

## Completion evidence

A valid result needs device-free positive and negative tests at both accepted widths, one stable execution identity, matched B=1 versus B=8 physical roots with an eight-prompt corpus, exact output parity, clean resets, and every predeclared bound passing. A rejected receipt that names the failing bounds also completes the change truthfully.

## False completion cases

Layer-level or full-model curve data alone, extrapolation from B=4, unmatched identities, warm-up samples counted as measured, or a weakened bound do not complete the goal.

## Audit risks

Service-level aggregate may sit near the 6.0 bound; eight serialized prefills amplify any per-user prefill regression; trace capture at width 8 is larger and may approach the capture timeout; identity drift between roots invalidates the receipt.

## Budget

One candidate, one device-free build, one bounded physical outage with mandatory production restoration, matched `single,batch,batch,single` order, two roots per route. Allowed outcomes are validated, blocked, exhausted, or user-decision-required.

## Approach registry

| Family | Mechanism | Evidence | State |
| --- | --- | --- | --- |
| same-width trace reuse | Preserve one trace across groups | Rejected on physical P150x2: parked-trace prefill tax 1.27 s exceeds the 0.85 s capture saving. | falsified |
| wider batch admission | Formalize width-8 serving already executed by arrival | Full-model curve: 106.35 t/s aggregate, retention 0.798, tail 1.25, all above 75-percent bounds. | active |
| prefill cost | Eliminate per-user host GDN snapshot round trips | Attribution blocked on host dispatch gaps; needs model-patch-level change. | blocked |
| B=16 admission | Widen to 16 | Curve efficiency 65 percent is below the 75-percent floor; retention 0.65 fails. | falsified |

## Adversarial check

The control must reject a width-9 profile, reject roots with slice gaps, and preserve the exact rejection if the aggregate ratio lands below 6.0.
