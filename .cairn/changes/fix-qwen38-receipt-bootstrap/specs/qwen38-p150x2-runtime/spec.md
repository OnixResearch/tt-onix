## ADDED Requirements

### Requirement: A receipt binding is not an execution identity input

r[tenstorrent.qwen38_p150x2.receipt_bootstrap]

The Qwen execution identity MUST exclude mutable receipt bindings. Immutable
policy, contract, validated-policy, and profile-source text MUST remain identity
inputs, and a change to any of them MUST move the identity. A package build MUST
carry an admitted exported admission profile that names the receipt the package
carries. A receipt MUST name an execution identity that a build of this tree
reproduces; a receipt whose named identity no build reproduces MUST NOT be
presented as carried by a servable package, and its admission MUST be re-earned
on a buildable identity.

#### Scenario: The binding returns to the identity

- GIVEN the execution identity input list
- WHEN any input name contains `-binding`
- THEN evaluation fails closed, naming the binding input.

#### Scenario: Immutable text moves

- GIVEN the shipped v3 policy input
- WHEN it is replaced by a fixture with different bytes
- THEN the execution identity store path differs.

#### Scenario: Receipt-only promotion

- GIVEN the shipped v3 profile, admitted with the committed receipt
- WHEN the same immutable v3 text is re-exported with the binding reverted to `not-measured`
- THEN the two exports differ exactly in `physically_admitted` and
  `identities.receipt_blake3`, which are `true`/the committed receipt BLAKE3 and
  `false`/`"not-measured"`, and the execution identity manifest carries no
  binding input.

#### Scenario: The receipt names an unreproducible identity

- GIVEN a receipt that names an execution identity no build reproduces
- WHEN a servable package is described
- THEN the receipt is not claimed for that package, and a re-earn on a buildable
  identity is required before the admission is carried.
