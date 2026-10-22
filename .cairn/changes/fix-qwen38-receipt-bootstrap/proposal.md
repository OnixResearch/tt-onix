## Why

The B=8 v3 receipt could not be carried by any package this tree builds.

The v3 mutable receipt binding was one of the execution identity inputs. A build
made while the binding says `not-measured` therefore had a different execution
identity from the build made once the receipt is admitted. A receipt earned on the
first identity cannot name the second, so promoting the receipt into the binding
moved the identity out from under it: the promotion was unsatisfiable.

The stored evidence contradicts the validation prose's package association. The
receipt names execution identity `l4arf461…`, which no current build reproduces;
all 192 saved roots report that identity with profile
`qwen38-p150x2-batched-throughput-b8-v3`, while the served package `wlsl6…` exports
the generic B=8 profile as `…-b8-v2`; the campaign package named by the prose embeds
`4115hgb…`. Details:
`evidence/identity-mismatch-2026-10-22.md`.

## What Changes

- Remove the v3 receipt binding from the execution identity inputs, and make that
  exclusion fail closed: an assert rejects any `-binding` input.
- Keep a positive structural check that the identity still moves when immutable
  policy text moves, and that the identity manifest carries the v3 policy,
  contracts, validated policy, and profile sources.
- Prove the receipt-only promotion exactly: re-export the same v3 policy and
  profile with the binding reverted to `not-measured`, delete exactly
  `physically_admitted` and `identities.receipt_blake3` from both exports, and
  require the remainder equal, with the admission fields false/`not-measured` and
  true/the committed receipt BLAKE3.
- Correct the documentation that read the v3 receipt as carried by a served
  package, and state that the v3 admission must be re-earned on a buildable
  identity (`evidence/re-earn-procedure-2026-10-22.md`).

## Impact

- **Files**: `pkgs/qwen36/default.nix`, two new fixtures under
  `pkgs/qwen36/fixtures/`, the validation and deployment documents, `README.md`,
  and this change directory.
- **Testing**: the Qwen package build and its check, plus the repository quality
  artifact and Cairn gates. No hardware, no simulator, no campaign.
- **Risk**: the execution identity moves relative to any build that included the
  binding. That is intended and is why no existing admission is claimed for it;
  the receipt's numbers are untouched.
