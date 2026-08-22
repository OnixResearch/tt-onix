## Why

The pinned P150x2 runtime cannot use the fused GDN chunk operation. Its 128-token masked fallback failed token parity, hybrid-state equivalence, stability, and target-verification economics. Before the repository loads MTP weights, it needs a bounded native verifier that handles widths 1, 2, and 4 without pretending that repeated full-model decode is amortized verification.

## What Changes

- Evaluate package-matched and pinned-runtime mechanisms for one short-width hybrid forward.
- Add the smallest diagnostic verifier that batches weight-bearing work while preserving ordered GDN recurrence and paged full-attention continuation, if the pinned runtime supports it.
- Extend physical evidence to cover per-position target logits, GDN recurrent and convolution state, full-attention KV writes, timing stability, and explicit non-admission.
- Return an exact source-level blocker when every bounded pinned-runtime route is invalid, rather than changing the runtime pin or production mode.

## Impact

- **Files**: Qwen36 model compatibility patches, diagnostic probe adapters, typed probe policy, tests, evidence, adoption documentation, and the Qwen38 P150x2 runtime specification.
- **Testing**: Baseline and post-change Python checks, positive and negative policy tests, package checks, focused Nix quality gates, and one bounded physical P150x2 pilot only after device-free validation.
