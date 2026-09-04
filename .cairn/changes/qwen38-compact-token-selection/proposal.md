## Why

The persistent MTP verifier reads full vocabulary logits and performs host argmax. Existing plain decode already uses shard-local winner reduction. Sampled serving also has a measurable selection-path cost.

Finding 2 in `docs/research/qwen38-p150x2-2026-09-04.md` identifies the exact source and limits.

## What Changes

- Add a compact verifier tail that returns per-row winners without a full-vocabulary gather.
- Evaluate bounded sub-shard top-K reduction for the sampled route.
- Preserve global token IDs, deterministic ties, and the existing sampling distribution.
- Retain full-logit output only for explicit diagnostics.

## Impact

The selection core owns merge semantics. TTNN adapters own reduction and readback. Reuse the existing decode reduction and `sampled_decode_core.py` rather than a second sampler.

The verifier change depends on the existing MTP lifecycle repair. The sampled candidate has separate timing and quality gates. This change does not enable sampled MTP, alter the first prefill token, or change production defaults.
