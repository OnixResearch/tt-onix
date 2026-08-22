## Why

Traced decode is much faster than eager decode, but the remaining device cost has only a static weight-traffic estimate. GDN, MLP, full attention, LM-head reduction, input update, and bounded readback have not been measured as separate replay components on the exact P150x2 package.

## What Changes

- Add a diagnostic-only component replay probe with a typed Nickel profile.
- Measure repeated traced replay for representative GDN mixers, MLPs, full-attention mixers, embedding, and LM-head reduction.
- Measure matched full-model replay, synchronized input update, and bounded winner readback.
- Use a deterministic core to reject unstable samples or an additive model that does not close against the full trace.
- Preserve eager production and select no optimization when the evidence is incomplete.

## Impact

- **Files**: Qwen diagnostic core, TTNN adapter, command wrapper, typed profile, tests, evidence, and adoption documentation
- **Testing**: positive and negative core tests, package gates, focused repository-quality gates, and one bounded physical P150x2 run
