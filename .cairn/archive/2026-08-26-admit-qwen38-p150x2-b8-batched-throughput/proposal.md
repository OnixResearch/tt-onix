## Why

The deployed P150x2 coordinator already serves arrival-driven batches up to width 16, but only B=4 carries a matched physical admission receipt. The measured full-model curve puts B=8 at about `106` aggregate tokens/second, `6.4x` single-stream, with per-user retention about `0.80`. B=8 serving therefore exists today without formal evidence, and the B=4 receipt explicitly does not cover it.

## What Changes

- Add a closed B=1 versus B=8 batch-admission profile with its own immutable policy and mutable receipt binding.
- Keep the B=4 profile, contracts, binding, and receipt untouched.
- Parameterize the pure admission core and Rust receipt evaluator over the profile-declared candidate width, accepting only widths 4 and 8.
- Package a dedicated `qwen36-p150x2-b8-admission` evidence entry point.
- Run matched physical roots in exact `single,batch,batch,single` order on devices 0 and 1 with an eight-prompt corpus.
- Admit only if every predeclared gate passes. Preserve an exact rejected receipt otherwise.

## Impact

- **Files**: `pkgs/qwen36`, `scripts/qwen36-batch-receipt.rs`, the Qwen runtime spec, and physical evidence.
- **Testing**: Package checks, positive and negative profile tests, receipt self-tests at both widths, Cairn gates, deterministic profile export, matched physical roots, and production restoration.
