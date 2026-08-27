# Search frame

## Goal

Reduce per-request host staging cost on the admitted B=4 served path without changing outputs.

## Completion evidence

Device-free gates pass on the patched package; a fresh matched `single,batch,batch,single` B=4 admission receipt on the new execution identity passes every predeclared gate, or the exact rejection is preserved. State-install and TTFT phase medians from the receipt quantify the gain.

## False completion cases

Source markers without a passing receipt, warm-up samples counted as measured, transferring the prior receipt across the identity change, or a weakened bound.

## Audit risks

Clone aliasing under a resident trace (bounded by the gate), write_slot consuming persistent scratch instead of clones, deferred logits holding device memory, RoPE table drift from the replaced trig, identity drift between roots.

## Budget

One patch, one package build, one bounded physical outage with mandatory production restoration, four matched roots. Allowed outcomes: validated, blocked, exhausted, or user-decision-required.

## Approach registry

| Family | Mechanism | Evidence | State |
| --- | --- | --- | --- |
| same-width trace reuse | Keep one trace across groups | Rejected: parked-trace prefill tax 1.27 s exceeds the 0.85 s capture saving. | falsified |
| B=8 admission | Formalize width-8 serving | Rejected at 73.5 percent efficiency; decode-side, not addressed here. | falsified (this change does not retry it) |
| device-resident staging | Remove per-user host round trip; cache RoPE tables | B=4 state install 0.494 s median; upstream write_slot already accepts device sources. | active |
| batched prefill kernel | One forward for all users | GDN chunk-seq kernel caps at B<=2 on TP=2 (BH<=32); not serviceable on 0.74. | blocked |

## Adversarial check

The control must fail closed if clones alias a resident trace, if write_slot consumes the persistent scratch, or if RoPE table rows differ bitwise from the replaced per-step trig.
