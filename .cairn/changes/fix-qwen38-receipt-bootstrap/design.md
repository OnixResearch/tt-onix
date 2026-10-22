## Decisions

### Decision: the bootstrap rule — a receipt binding is not an identity input

**Choice:** the v3 receipt binding is exported beside the profile but is not one of
`executionIdentityLocalInputs`. An assert fails closed if any identity input name
contains `-binding`. The immutable v3 policy, validated policy, contracts, and
profile sources stay in the identity.

**Rationale:** a binding is a mutable pointer to evidence, not policy text. If it
were an identity input, the identity would move exactly when the receipt is
admitted, so no build could carry its own receipt. This is the smallest rule that
makes promotion satisfiable without letting mutable evidence into the identity.

### Decision: the structural identity proof, and its residual assumption

**Choice:** the check asserts (a) no `-binding` input is in the identity input
list, and (b) `mkExecutionIdentity` reads only that manifest. The policy-drift
assert swaps the v3 policy for a fixture with different bytes and requires a
different identity store path. The install check additionally greps the shipped
manifest for the v3 policy, contracts, validated policy, and profile sources.

**Rationale:** this is structural, not a measured two-build comparison of identity
paths. The install check re-exports the profile in the same build directory and
uses the same manifest; it does not build a second package. The residual assumption
is that `mkExecutionIdentity` has no input outside the manifest — its definition
reads only `executionIdentityInputs`, and the policy-drift assert exercises that
path. A measured two-build comparison remains the stronger evidence and is left to
the re-earn gate in `evidence/re-earn-procedure-2026-10-22.md`, which requires
`identity(binding=not-measured) == identity(binding=admitted)` on two outputs.

### Decision: the exact two-field profile proof

**Choice:** `installCheckPhase` exports the shipped v3 profile and a re-export of
the identical immutable v3 text with the binding fixture reverted to
`not-measured`. A deterministic Python step loads both JSON objects, captures
`physically_admitted` and `identities.receipt_blake3`, deletes exactly those two
fields from both, and requires the whole remainder equal. It requires the shipped
fields to be `true` and the committed receipt BLAKE3 and the reverted fields to be
`false` and `"not-measured"`. Any other difference prints a unified diff of the
remainder and fails.

**Rationale:** byte equality of two unrelated JSON exports proves too little and
too much: it would reject key-order or serialization differences that do not move
policy, and it would accept any arbitrary difference. Deleting exactly the two
admission-derived fields and comparing the rest states the promotion contract
directly.

### Decision: re-earning is required, because the identity moved

**Choice:** no existing admission is carried forward. The documentation says the
v3 admission must be re-earned on a buildable identity, following
`evidence/re-earn-procedure-2026-10-22.md`.

**Rationale:** adding the v3 policy, contracts, validated policy, and profile
sources to the identity inputs, and removing the binding, both moved the identity.
The stored receipt names `l4arf461…`, which this tree does not reproduce, and the
served package exports `…-b8-v2`. Even if the structural bootstrap rule is now
correct, the historical receipt's numbers do not name a buildable identity.

## Risks / Trade-offs

- The residual structural assumption above: a build could add an identity input
  outside `executionIdentityInputs` only by changing `mkExecutionIdentity`; the
  fail-closed assert and the policy-drift assert would then be reviewed in the same
  diff.
- A receipt-only promotion now changes only the exported profile, so a promoted
  package is a different package store path with the same execution identity. The
  re-earn procedure still requires the wrapper identity to match the measured one.
