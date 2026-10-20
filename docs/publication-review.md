# Publication review (not release approval)

This checkout is **not cleared for public release**. Do not change the
repository visibility, push another ref, publish a source archive, or
distribute a build output based on this checklist alone. This review covers
the working branch as inspected on 2026-10-12; `origin/main`, other branches
and tags require their own release-boundary review. Nothing here changes an
evidence receipt, Git history, or a deployed service. Local license metadata
now matches the existing LGPL-2.1 root license; upstream rights are unchanged.

## What the local audit established

- At the start of this review, `HEAD` had 2,136 tracked files (about 28 MB). `.pi/` and top-level `generated/` are ignored now, but ignoring a path does not erase committed versions.
- A read-only scan of **all locally reachable refs** enumerated 7,859 Git objects, including 3,787 blob objects. A historical `generated/inspector/mesh_workloads_log.yaml` blob is about 72 MB even though the file is no longer tracked. Forty `.gz` blobs were decompressed and pattern-scanned. No private-key PEM header, common GitHub/Hugging Face/OpenAI/AWS token format, or literal bearer authorization header was found by those patterns. This is not a credential or privacy clearance: unusual encodings, other providers, arbitrary JSON fields, binary payloads, and copies outside local refs were not comprehensively reviewed.
- Current files and historical blobs contain personal home paths, machine names,
  operational details and raw logs. Current examples include
  `docs/evidence/tt-metal-0.77/qwen-binding-identity/empty-corpus.log`,
  `pkgs/qwen36/evidence/2026-09-30/qwen38-two-long-prefix-candidate.json`, and
  compressed logs under `docs/evidence/tt-metal-0.77/` and
  `.cairn/changes/optimize-gdn-decode-kernel-grid-and-prep/evidence/`.
  `scripts/qwen38-profile-service` and `scripts/qwen38-profile-99.conf` were
  removed from the Git index but preserved on this owner machine; earlier
  versions, including the hardcoded paths in `scripts/qwen38-step-probe.sh` and
  `nixos/examples/qwen38-p150x2-read-lanes.nix`, remain in reachable history.
  The device-4 runbook names a local SSH identity **path** and includes a host
  fingerprint; its saved `known_hosts` fixture contains a **public** key, not a
  private key. Both still identify a real host configuration.
- Receipts, binary transports and two `adapter.safetensors` fixtures are tracked. An evidence digest establishes byte identity, not permission to publish private conversations or model-derived weights. The captured request/response content and binary payloads need human review. The artifact gate in `docs/repository-quality.md` checks only narrow private markers in evidence; it does not certify publication safety or audit Git history.
- `LICENSE` retains the full LGPL-2.1 text. Thirteen repository-owned Cargo
  declarations and fifteen local Nix output labels previously claimed MIT;
  they now say `LGPL-2.1-only`. This aligns current metadata but does not
  retroactively relicense past revisions or establish ownership of imported
  work. Apache-2.0-marked TT-Metal/Qwen patch portions and a ttWKV7 descriptor
  fixture retain their upstream-style notices and a separate full license
  text. `pkgs/ttwkv7/runtime.nix` correctly marks its pinned upstream unfree:
  no upstream license was declared. Publishing its Nix recipe is a separate
  rights question from distributing the upstream source or binary.
- A fresh public Nix build is not yet established. `flake.nix` directly pins `OnixResearch/cairn`; `flake.lock` also pins `OnixResearch/onix-artifact` through an SSH-only URL and transitively pins `OnixResearch/slotcar`. Anonymous GitHub API requests for those three repositories returned 404 on 2026-10-12 (private, removed, or otherwise unavailable anonymously); the local Nix cache can mask that problem. The pinned `OnixResearch/nickel-export` repository was visible anonymously. Publish the required dependencies at the pinned revisions or replace them with publicly fetchable equivalents before claiming a reproducible public flake.
- Device-free `nix run .#quality` now evaluates both Linux systems and reaches
  the aggregate build with ignored `generated/` present. The focused Python,
  Nix-format, and shell checks pass after mechanical formatting; AST shapes
  remained identical across the 27 formatted Python files. The aggregate
  still fails the artifact-manifest entry count check (**375 expected, 624
  found**). The chosen public boundary omits all current captured evidence;
  leave the private manifest and allowed roots unchanged rather than treating
  unreviewed files as cleared. Any later approved candidate must update
  ownership, hashes, and consumers without misrepresenting physical results.
  Qwen source formatting changes package execution identity; the
  read-lane example deliberately rejects its new output until physical
  requalification. A device-free pass does not qualify it for serving.
- `nix build .#checks.x86_64-linux.qwen36 --no-link` rebuilt the changed Qwen
  package and passed its device-free import, service-contract, structured-read,
  and other install checks. The focused
  `nix build .#checks.x86_64-linux.qwen36-nixos-service --no-link` check also
  passed. These use locally available inputs and do not
  validate anonymous dependency fetches, recurrent-state silicon behavior, or
  the new package as a deployed host owner.
- Git commit author names and email addresses are part of the reachable history. Obtain the contributors' consent to publishing those identities or use an appropriately reviewed history strategy. Do not assume removing a file in a new commit hides earlier commits.

## Chosen boundary for a future public candidate

The current release set is empty: no branch, tag, Git history, source archive,
build output or saved evidence in this checkout is cleared for publication.
Keep the existing checkout and its history private. A future candidate must
be a separately reviewed clean-history source export, not a visibility change
or a push of any current ref. Preserve attribution and license obligations
even when omitting Git history.

By default, omit all 616 files currently classified as `portable_evidence`
(including the 249 new entries), not just raw and compressed logs. None has
been cleared for a public export by this review. The seven generated-source
entries, the large fixture, and other binary or model-derived artifacts are
not automatically cleared either. Re-admit only individually reviewed,
publishable files or truthful redacted summaries; never fabricate or alter a
physical result. `pkgs/qwen36/default.nix` binds B=8/B=16 admission profiles
to saved receipt hashes, so a source-only export must revise those checks
and their claims together. Keep the private originals unchanged.

The exact Git-filtered quality source has 249 added evidence entries and no
removed or changed entries: 154 under `docs/evidence`, 35 under `pkgs/qwen36`,
33 under `.cairn/changes`, 26 under `.cairn/archive`, and one under
`docs/research`. All 154 `docs/evidence` entries are also outside the
portable-evidence allowed roots in `quality/generated/repository-quality.json`.
Do not regenerate the manifest, expand that allowlist or call the gate green
to make this private working tree look release-ready.

## Work required before publishing

1. **Implement the selected boundary.** Prepare a separate clean-history
   source export from reviewed files; do not push current refs or include
   private blobs merely because they are committed. Preserve attribution and
   license notices. Inventory tracked, deleted and historical blobs in a
   private staging clone (`git ls-files`, `git rev-list --objects --all`),
   including compressed archives and the historical generated log. Re-scan
   the exact export before making any ref, archive or output public.
2. **Classify disclosure and remediate at the source.** Review every retained hardware log/receipt, SSH metadata, host-specific runbook, model output and binary fixture with the owner. Retain only approved portable evidence; otherwise replace it with a truthful sanitized summary or omit it from the release. If evidence changes, update its manifest and any consumer tests/docs, preserving the distinction between an original physical result and a redacted account. Re-scan the exact release tree and all release-bound refs afterward. If a real credential is found, revoke/rotate it immediately; deleting a commit is not rotation. If a historical blob cannot be public, use an owner-approved clean-history export or rewrite **every published ref and tag** containing it before publication. Do not force-push this operational repository as an incidental cleanup.
3. **Resolve rights and dependencies.** Confirm that the LGPL-2.1 grant covers
   retained `onix-core` contributions and reconcile historical MIT-labeled
   revisions, original patches and Apache-2.0-marked upstream portions.
   Changing package metadata does not grant rights to external code. Check
   pinned upstream terms and notices at their exact revisions. Decide whether
   ttWKV7 can be referenced, built or distributed in each intended channel;
   do not infer permission from `allowUnfree` or a URL. Verify provenance and
   redistribution terms for the two saved LoRA adapters and the recorded
   transport transcripts. Make every locked input fetchable without private
   credentials, including Cairn and its transitive dependencies, then verify
   in a clean store with anonymous access. If distributing binary outputs,
   audit their actual third-party source, submodules, model assets and notices
   separately from this source-tree review.
4. **Prepare the public process.** Enable a private vulnerability reporting route on the hosting repository, choose a maintainer contact and review `CONTRIBUTING.md`. Validate the candidate *as published*: run `nix run .#quality` on the reviewed source tree, check public README links and package references, and exercise at least one supported device-free package. Hardware results from historical evidence do not become fresh public acceptance results. Record the chosen release commit and approvals outside the source tree.

`.gitignore` now excludes common new secret files, local model directories, and
the two owner-only migration tools. Those tools remain in the local checkout
but are removed from the Git index. Ignoring or untracking does not remove
historical objects, inspect binary content, or make existing refs safe to
publish.
