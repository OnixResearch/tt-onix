# Device-free validation for the sampled-mode admission change

Run on 2026-08-27 at the repository root, before any commit of this change.

## Pure core probe

`python3` driver loading `pkgs/qwen36/check.py` and running only
`probe_kernel_distribution_admission`:

- `PASS kernel distribution admission`
- positive: close candidate admitted, eligible modes `("sampled",)`,
  near-tie positions stratified and reported
- negative: stable-position argmax flips rejected on the agreement floor;
  flattened top-K probabilities rejected on the TV bound; disjoint top-K sets,
  a four-prompt corpus, an unbalanced root order, a non-teacher-forced probe
  mode, and non-finite probabilities all rejected fail-closed

## Nickel policy validation

`nickel export kernel-distribution-admission-policy-validated.ncl` in
`pkgs/qwen36` (via `nix run nixpkgs#nickel`, nickel 1.17.0) exports the
closed policy through `PolicyClosed`, `PolicySemantic`, and `Policy`.

## Cairn strict validation

Both trees carry this change byte-identically.

- Pinned CLI `github:OnixResearch/cairn/837d7307...#cairn` with the policy
  shipped in that source, `validate --root . --strict`:
  `valid: true`, 16 specs validated, 1 active change
  (`qwen38-gdn-sampled-mode-admission`), no issues.
- The installed CLI in the profile cannot parse either the pinned or the
  `15f0087` policy (`lifecycle_store_policy.project_config.required_fields`).
  This is the pre-existing policy-version mismatch recorded with the Cairn
  migration; native `.cairn`-only validation waits for the pin uplift (task
  A6).

## Not claimed

- No device probe, screen, or physical evidence exists yet. Phase 2 and
  Phase 3 tasks own that work.
- The fused GDN candidate stays default-off. Nothing in this change changes
  production routing, admission, or deployment.
- Adding the policy files to `executionIdentityLocalInputs` intentionally
  changes the package execution identity; previously measured receipts stay
  bound to the older identity, so a future deployment needs a fresh receipt
  under the existing admission requirements.
