# Tasks: Keep the receipt binding out of the execution identity

## Phase 1: Implementation

- [x] [serial] I1 Remove the v3 receipt binding from the execution identity inputs, add the fail-closed assert that no `-binding` input is an identity input, add the policy-drift identity assert with its fixture, and keep the v3 policy, contracts, validated policy, and profile sources in the manifest. r[tenstorrent.qwen38_p150x2.receipt_bootstrap]
- [x] [serial] I2 Prove the receipt-only promotion exactly: re-export the v3 profile with the binding reverted to `not-measured`, delete exactly `physically_admitted` and `identities.receipt_blake3` from both exports, require the remainder equal, and require the named admission fields on both sides. r[tenstorrent.qwen38_p150x2.receipt_bootstrap]
- [x] [serial] I3 Correct the validation prose and the served-deployment documents: the v3 receipt names a served identity that no build reproduces, is not carried by a servable package, and must be re-earned on a buildable identity. r[tenstorrent.qwen38_p150x2.receipt_bootstrap]

## Phase 2: Verification

- [x] [serial] V1 `nix build .#qwen38 --no-link -L` passes; the package's install check prints that the promotion changed exactly the two admission fields. r[tenstorrent.qwen38_p150x2.receipt_bootstrap]
- [x] [serial] V2 `nix build .#checks.x86_64-linux.qwen38 --no-link -L` passes. r[tenstorrent.qwen38_p150x2.receipt_bootstrap]
- [x] [serial] V3 `nix build .#checks.x86_64-linux.repository-quality-artifacts .#checks.x86_64-linux.repository-quality-cairn --no-link -L` passes with this change's fixture and Cairn files as the only manifest additions. r[tenstorrent.qwen38_p150x2.receipt_bootstrap]

## Phase 3: Completion

- [x] [serial] D1 Record the identity mismatch and the predeclared re-earn procedure under `evidence/`, making no claim that a campaign or a new admission was run. r[tenstorrent.qwen38_p150x2.receipt_bootstrap]

## Phase 4: Re-earn on the buildable identity

- [x] [serial] R1 Run the predeclared eight-root interleaved B=8 campaign on the tree's own package with the serving route (`QWEN35_GDN_DECODE_FUSION=step`, grouped and traced prefill), recording per-root provenance, all 192 measured responses and the frozen launch environment. r[tenstorrent.qwen38_p150x2.receipt_bootstrap]
- [x] [serial] R2 Build the receipt from those responses with the unmeasured twin as the input profile and validate it independently: the six gates and every structural gate pass, `admitted = true` with no reasons. r[tenstorrent.qwen38_p150x2.receipt_bootstrap]
- [x] [serial] R3 Move the binding and the pinned receipt to the fresh digest, rebuild device-free, and prove the promotion gate: promoted execution identity equals the measured identity and the profile delta is exactly `physically_admitted` plus `identities.receipt_blake3`. r[tenstorrent.qwen38_p150x2.receipt_bootstrap]

## Phase 5: All admissions on the promoted package

- [x] [serial] R4 Re-earn the scheduler and prefix-cache admissions on the promoted package `gzwghi7…`/identity `qwnxwc…`, then verify the deploy guard read-only: both required receipt kinds are found and containment passes, so `scripts/qwen38-deploy` accepts it without `--force`. The deploy move itself was not executed. r[tenstorrent.qwen38_p150x2.receipt_bootstrap]  (evidence/all-admissions-promoted-2026-10-26.md)
