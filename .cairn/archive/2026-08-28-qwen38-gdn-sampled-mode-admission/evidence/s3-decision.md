# S3 owner decision (2026-08-27)

The owner reviewed the measured screen table in `s2-screen-result.md` and
selected **option 3**.

## What changes

- The per-position minimum logit PCC floor (0.99) is removed as an admission
  gate. The core, the Nickel policy, the contracts, and the spec delta drop
  `minimum_min_logit_pcc`.
- The receipt keeps reporting `min_logit_pcc` as a diagnostic metric.

## What stands unchanged

- mean shared-token logit PCC floor 0.995
- mean top-K TV cap 0.10, maximum top-K TV cap 0.25
- mean top-K residual mass cap 0.50
- near-tie margin epsilon 0.5, stable-agreement floor 0.95 over at least 120
  stable positions
- corpus floor 5 prompts x 48 positions, root order
  `control,candidate,candidate,control`, K = 20
- state isolation, clean reset, GDN ratio 0.95, full-step ratio 0.97
- sampled-only eligibility; greedy and deterministic serving keep exact
  parity

## Rationale

The screen rejected only on the minimum PCC floor. Nine of 240 positions
fell below 0.99, five of them at near-tie margins where shared-top-20 tail
ordering destabilizes Pearson correlation. Every TV bound and all 220 stable
argmaxes passed. Total variation is the direct distributional bound; the
minimum-PCC floor was the heuristic that near-tie tails made noisy.

## Consequence

The policy revision changes the execution identity. Before any admission the
package rebuilds and both matched screens re-run under the new identity
(task S3a), then fresh B=4 admission runs for the changed identity (task D1).
