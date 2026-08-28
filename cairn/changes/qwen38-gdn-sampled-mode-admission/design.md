## Context

The archived change `2026-08-27-fuse-qwen38-gdn-decode-state-update` exhausted its exact-parity search. The fused fp32 recurrence keeps state PCC near 0.99997 over two steps, but the composite reference unpacks FP32 matmul inputs to TF32 inside the mixed kernel, and the fused binary-SFPU route is more precise, not less. The drift accumulates through 48 GDN layers and crosses late argmax margins: prompts diverged at generated indices 47, 32, and 16, and one prompt stayed exact for all 64 tokens. Speed, state isolation, and reset gates all passed.

The service samples with host-side temperature on every non-greedy request. Two runs of the admitted production path already emit different tokens for the same prompt. Exact token parity between control and candidate is therefore not the user-visible contract for sampled traffic; closeness of the token distribution is.

Field practice: the Qwen3.6 bring-up gates per-module PCC; the Adartras bundle gates on PCC plus invariant tests plus chi-squared sampling checks and still declares its BFP4 accuracy cost unquantified; upstream tt-metal 0.77 and llama.cpp PRs 26111/25601 ship fused linear-attention kernels with no exact-token requirement. This repository holds the strictest bar in the set. That bar is kept where it is observable and correct: greedy and deterministic serving.

## Decisions

### Decision: Scope the weaker gate to sampled serving

**Choice:** A distributional admission receipt can make a candidate eligible for `sampled` requests only. Greedy and deterministic requests keep the exact-parity path and the previously admitted identity.

**Rationale:** Deterministic replay keeps its strongest guarantee. Sampled serving gets a contract that matches what users can observe: a near-identical distribution, not identical tokens.

### Decision: Gate on four bounded metrics, all owned by Nickel

**Choice:** Teacher-forced probe evidence on matched physical roots in exact `control,candidate,candidate,control` order, gated by:

1. Union logit PCC per position: mean at least 0.995, minimum at least 0.99.
2. Truncated top-K (K=20, the model's recommended sampling top_k) total variation: mean at most 0.10, maximum at most 0.25.
3. Mean top-K residual mass at most 0.50, so truncated comparisons stay meaningful.
4. Margin-stratified argmax agreement: positions whose control top1-top2 logit margin is at least 0.5 must keep their argmax with agreement at least 0.95 over at least 120 stable positions. Near-tie positions below the epsilon are reported but never gated.

State isolation, clean reset, the GDN ratio bound of 0.95, and the full-step ratio bound of 0.97 carry over unchanged from the rejected change.

**Rationale:** PCC and TV bound distributional closeness directly at every position. Margin stratification separates the two failure classes the root cause found: drift that flips a near-tie (harmless to a sampled endpoint) and drift that flips a confident argmax (a real distribution change). Every constant is owner-reviewable and any change to one is an execution-identity change.

### Decision: The current fused candidate is not presumed to pass

**Choice:** The gate is fixed before the screen runs. The archived 64-token evidence implies roughly 62 percent raw agreement over four prompts, and the stable-position fraction under a 0.5 epsilon is unknown until the probe measures it.

**Rationale:** Tuning a gate to pass an existing candidate would reproduce the failure mode this repository exists to prevent. If the candidate fails the stable-agreement bound, the follow-up is the staged precision-preserving kernel design the root-cause search named, not a weaker gate.

### Decision: Keep both Cairn trees in lockstep until the pin uplift

**Choice:** Mirror this change into both `cairn/changes/` and `.cairn/changes/`. Retire the legacy tree only after the flake pin moves to a Cairn revision that reads the native `.cairn` layout.

**Rationale:** The pinned CLI (`837d7307`) reads only `cairn/` and drives the repository-quality nix gate, so removing the legacy tree now empties that gate. The installed CLI at `15f0087` and its policy read the native `.cairn` tree. `diff -rq` confirms the trees differ only by this change directory, so mirroring keeps them byte-identical.

## Risks / Trade-offs

- Threshold constants (0.995, 0.99, 0.10, 0.25, 0.50, 0.5, 0.95, 120) are engineering judgments anchored to field practice, not measured model data. Task S3 requires owner review of every constant before any admission.
- Sampled-only routing adds a serving-mode condition to the scheduler. It is not implemented in this change; a separate task wires it after a passing screen.
- Greedy users keep the slower path by design. That is the cost of keeping the deterministic guarantee.
- A passing distributional receipt still supports no accuracy-benchmark claim. The non-claims list must say so, matching the Adartras posture of declaring precision costs unquantified.
