# Drain pass 2026-08-27: Phase 2 progress and blockers

Worktree: `/home/brittonr/git-worktrees/qwen38-gdn-sampled-drain-20260827`
on branch `qwen38-gdn-sampled-mode-admission-drain-20260827` from
`origin/main` at `27013e2`.

## Gates

Canonical Cairn CLI at `~/git/OnixResearch/cairn` (`15f0087`):
`gate proposal`, `gate design`, and `gate tasks` all PASS for this change
before implementation.

## P1: pure probe builder (complete)

New `qwen36_service/kernel_distribution_probe_core.py`:

- `PositionReadback` validates full-vocab logit readbacks (unique integer
  tokens, finite logits, vocabulary strictly larger than the top-K cut).
- `build_position_sample` computes full-softmax probabilities, selects each
  side's top-K with a deterministic tie-break, and reuses the admission
  core's evidence types.
- `canonical_corpus_payload` gives the shell a deterministic serialization;
  the shell hashes it with `b3sum` and the core validates the 64-hex digest
  before it can enter a receipt. The pure core computes no digests, matching
  the house pattern (`b3sum` at the package level, cores validate hex).
- `evaluate_probe` emits the deterministic receipt with every metric,
  eligible modes, rejection reasons, and the corpus digest.

Device-free check additions (`check.py`): positive receipt for a close
candidate, sampled-only eligibility, digest passthrough, canonical payload
determinism and drift sensitivity, plus negative cases for duplicate tokens,
top-K-sized readbacks, mismatched vocabulary lengths, non-finite logits,
short corpora, unbalanced root order, wrong probe mode, and an invalid digest.

Result: `PASS kernel distribution admission` and
`PASS kernel distribution probe builder` in the worktree.

The device readback shell (TTNN teacher-forced loop with per-position
full-vocab readback on paired roots) is deliberately part of task S2: it
cannot be exercised without the P150x2 screen.

## S1: package rebuild (complete)

`nix build path:<worktree>#qwen36` completed with exit 0 after a full
installCheckPhase (19m41s). The install test now also greps
`kernel-distribution-admission-policy.ncl` inside the execution identity
manifest, so the build itself fails if the new policy drops out of the
identity.

- package: `/nix/store/lk1agynj00jfa5sskdx7jb4x2wm4d6dk-qwen36-0.1.0`
- package_identity: `/nix/store/dfdgl49i7sa610z3s4j21lpn5wc81rx2-qwen36-execution-identity-0.1.0`

The candidate patch ships default-off inside this package; no candidate
selection is made. A pre-existing nanobind reference-count warning appeared
in the install check output and did not affect the result.

## Blockers (change stays active)

- **S2 (physical screen):** needs P150x2 hardware under authorized qwen38
  service control. Standing rule: never stop or mask the qwen unit without
  an explicit authorization for this screen. No such authorization exists in
  this drain request.
- **S3 (owner review):** every threshold constant (PCC 0.995/0.99, TV
  0.10/0.25, residual 0.50, margin 0.5, agreement 0.95, stable floor 120)
  requires owner sign-off against the measured screen before admission.
- **D1/D2:** depend on S2 and S3.

## Not claimed

- No probe receipt exists. No admission, routing, or deployment change.
- Production is untouched and the fused candidate stays default-off.
