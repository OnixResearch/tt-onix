## Context

Four runtimes share two P150 cards, and each one's launch contract lives in a
different place. The Qwen service is the only one with a unit, a port, and a
readiness endpoint. The other three are job-style commands: they take the cards,
run, and exit with a receipt.

The switch design that motivates this change (`docs/research/` notes and the step
plan in it) needs one machine-readable per-model record before it can add a
switch command, a record, or a privileged owner helper. This change delivers the
record and a read-only reader only.

Two existing registries matter:

- `model-targets/profiles.ncl` owns *model identity*: checkpoint artifacts,
  topology, state kind, precision, tolerances. Its `profile_identity` is BLAKE3
  over canonical JSON.
- `pkgs/qwen36/authority.ncl`, `pkgs/nanbeige43/generated/authority.json`, and the
  model-target artifact list each own *checkpoint admission*, in three different
  schemas.

Launch data is a third axis. It is recorded here and references those identities;
it does not replace them.

## Decisions

### Decision: A separate registry file, not an extension of `profiles.ncl`

**Choice:** Add `model-targets/runtime-profiles.ncl` beside `profiles.ncl`, with
its own contract, and reference `model_target_blake3` rather than re-deriving
model identity.

**Rationale:** `profiles.ncl` feeds `profile_identity`, and that value is pinned in
three places: `pkgs/rwkv7-p150x2/persistent-decode-profile.ncl:6`
(`admitted_model_profile_blake3 = d33ed105…`), `model-targets/check.nix:10`
(`expectedRwkvProfileIdentity`), and `model-targets/check.nix:180`, which asserts
the live identity equals the pin. Adding launch fields to the existing profile
objects would move every profile identity and force an avoidable re-admission of
the rwkv7 persistent-decode profile.

**Proof of non-change:** `model-targets/profiles.ncl` is
`3883a12a5daeb8c1578f356d7db2a31456783746d024d70ee13094da4df863de` and
`model-targets/generated/profiles.json` is
`88ad86c4d1ca4b879686a78638ddb7ebdbc771717274072b4f9862fc59216aa6`
before and after this change; `git status` reports neither modified, and the
qwen38 derivation path is unchanged. See `evidence/identity-invariance.md`.

### Decision: The registry is data, not policy

**Choice:** The profiles name admission *kinds* and the artifacts that implement
them; they never compute an admission verdict.

**Rationale:** The three admission mechanisms are genuinely different: a
two-receipt set keyed by `admission.admitted_profile.package`
(`scripts/qwen38-deploy:93-112`), a generated authority document with per-file
SHA-256 (`pkgs/nanbeige43/runner/nanbeige43_runner/authority.py:71-104`), and a
set of gate exits. A uniform "admitted" abstraction would hide which check
protects which risk. The profile records the references; the check verifies the
references exist, hash to the recorded value, and (for receipts) still say
admitted.

### Decision: `authority_blake3` names a document, and `authority_kind` names the
schema

**Choice:** For `generated-authority-blake3`, the profile also declares
`authority_path`; the check recomputes BLAKE3 over that file. For
`model-target-profile-identity`, the check compares against the live
`model-targets` identity.

**Rationale:** The Qwen package already treats `generated/authority.json`'s
BLAKE3 as `identities.authority_blake3`
(`pkgs/qwen36/decode-profile.ncl:27`); recomputing it makes that convention
verifiable instead of decorative. A field named `*_blake3` that held a SHA-256
would misstate its algorithm, so the nanbeige profile hashes its authority
document rather than its file digests, and the per-file SHA-256 admission stays
where it belongs.

### Decision: Explicit absence instead of sentinels

**Choice:** `port`, `checkpoint.path`, `checkpoint.authority_path`, `cache`,
`cache.key`, `precision.preset`, and `readiness.path` are optional, and a
cross-field rule decides when they must be present or absent.

**Rationale:** `rwkv7-ttnn` and `metalium-rs-rwkv7` take their checkpoint from a
Nix-materialized fixed-output derivation, not a host path; neither declares a
cache root. `""` would be a lie, and a sentinel like `"not-applicable"` would
collide with real values. `optional` is the contract's own absence, and the
semantics refuse a field that is present when it should be absent.

### Decision: `status` proves the record is rooted

**Choice:** The reader treats a record as authoritative only when its package
exists, its name matches the profile's `store_name`, and
`/nix/var/nix/gcroots/tenstorrent-models/<profile>` resolves to that exact path.

**Rationale:** The measured failure on this host is precisely a recorded path
that the garbage collector removed. A reader that reported "managed" from a
dangling record would make the same mistake as the symlink it replaces.

### Decision: Legacy sources are read, never trusted blindly

**Choice:** `status` also reads the operator symlink, the Nix profile generation
(including the packages its manifest installs), and the owner unit's state, then
reports a contradiction when they disagree with an authoritative record.

**Rationale:** During the transition three records coexist. Naming the
disagreement is the only way to cut over safely; silently preferring one is how
the current state became unreadable. The Nix profile's manifest is read directly
(`<generation>/manifest.json`), so the command needs no `nix` invocation.

## Risks / Trade-offs

- **A pinned receipt can be superseded.** The Qwen profile names two receipt
  files and their digests. A later admission must update this registry in the same
  change, or the check fails. That is deliberate: the registry must not drift from
  the evidence it claims.
- **Host paths are recorded.** `checkpoint.path` and `cache.root` for Qwen and
  nanbeige are host paths, exactly as the served drop-in and the recorded
  receipt use them. They are data for a launcher, not repository-relative
  references, and the check does not require them to exist on a build machine.
- **Two of four profiles declare no cache.** Absence is recorded as absence. A
  later cache-management change must add the field rather than assume a default.
- **`status` reads the systemd unit.** It uses `systemctl show`, which is
  read-only, and degrades to `unavailable` when no binary is present. It never
  stops, starts, enables, or masks anything, and its install check asserts the
  program text never names a device, `tt-smi`, or the owner helper.
- **The mask is reported, not repaired.** On this host the owner unit is masked.
  `status` reports that and exits zero when the record is absent and nothing is
  dangling; repairing a host generation is out of scope and out of privilege.

## Non-goals

- No switch command, no record writer, no GC-root creation, no owner helper, no
  reset, no service start or stop. Those are later steps.
- No change to `model-targets/profiles.ncl`, the model-target factory, any
  package, any receipt, or the NixOS module.
- No unification of numerical gates, receipt schemas, or precision presets.
