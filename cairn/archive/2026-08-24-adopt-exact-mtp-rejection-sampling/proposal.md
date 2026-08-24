## Why

Local MTP acceptance is greedy exact-match only. The drafter produces an
argmax; `accepted_prefix_length` counts only a draft that equals the target
argmax. If MTP runs for sampled requests, greedy acceptance biases the output
toward the argmax token.

The reviewed Adartras Qwen3.6-A3B bundle proves that, for an argmax
(point-mass) draft, standard speculative sampling reduces to rejection
sampling with acceptance probability equal to the target probability `p(x̂)`.
The emitted stream is then exactly target-distributed, and the per-position
realized divergence is exactly `|a − p(x̂)|`.

Adopting that decision core gives the Qwen3.8 MTP path a distribution-exact
sampled acceptance with a measured, not assumed, divergence.

## What Changes

- Add a pure, device-free MTP acceptance-rule core with exact, lenient,
  relaxed, and greedy rules, modeled on the reviewed bundle's proven core.
- Implement distribution-exact sampled speculative acceptance for the Qwen3.8
  MTP drafter path.
- Keep greedy exact-match acceptance as the production default.
- Report the realized per-position total-variation divergence for every rule.

## Impact

- **Files**: `pkgs/qwen36/service/qwen36_service` acceptance core and adapter,
  MTP drafter/selection wiring, schema, and device-free tests.
- **Testing**: positive and negative device-free pytest, Python compilation,
  Ruff, Qwen package checks, Cairn validation, and repository-quality checks.
