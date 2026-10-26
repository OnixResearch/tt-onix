## Context

`tt-model status` answers "what claims the cards", but it can only read a
source. It cannot say whether a candidate package would satisfy a profile, and
it cannot create the record it reads. Both belong in the same tool, because both
must fail closed on the same facts.

The record has to be provable later. A path in a JSON file is not proof: the
garbage collector can remove it, and this host already lost the package the
operator symlink names. The record therefore names only store paths it also
roots, and the reader checks the roots.

Privilege is deliberately absent. Everything here is a read of the store and the
repository, plus writes inside `TT_MODEL_STATE_ROOT` and `TT_MODEL_GC_ROOTS`. On
the host those directories are root-owned in the eventual deployment, so the
privileged owner helper in step 3-4 will call this writer rather than replacing
it. Nothing in this change stops or starts a unit, opens a device, or resets the
cards.

## Decisions

### Decision: `check` separates structural failures from admission failures

**Choice:** A failure whose item starts with `receipt-`, `authority-`, `gate-`
or `model-target` is an admission failure. Everything else is structural.
`record --force` may bypass admission failures only.

**Rationale:** A missing package, a missing entry binary, a missing runtime root,
or a checkpoint whose name does not carry the revision is a broken profile, not
an unadmitted one. Letting `--force` write such a record would create exactly
the dangling pointer this change exists to prevent.

### Decision: Registration is a query, and its absence is reported

**Choice:** `check` asks `nix path-info --offline --json` when a `nix` is
available, and otherwise accepts store presence and says so
(`method: store-presence`). `TT_MODEL_NIX=none` forces the weaker method, and
`TT_MODEL_NIX=<path>` pins the query.

**Rationale:** A materialized store path is a registered one; the store does not
keep invalid paths. The distinction is recorded rather than hidden, and the
install check can exercise both. A `nix` that runs and fails is still a failure,
which the install check proves with a fake binary.

### Decision: The checkpoint name must carry the revision

**Choice:** `checkpoint.path` must end with the full revision or its first 12
hexadecimal characters.

**Rationale:** The Qwen snapshot directory is named by the full revision; the
nanbeige snapshot directory is named `nanbeige4.3-33b-c1093c501cc6`, which
carries the short form. A rule that demanded the full revision would reject the
real checkpoint; a rule that demanded nothing would accept any directory.

### Decision: Deep re-hash is bounded and never implicit

**Choice:** `--deep` walks the checkpoint, hashes every file with `b3sum`, and
prints progress. Above 1 GiB it refuses before hashing unless `--allow-large` is
given. The shallow path stats nothing but the checkpoint's existence and name.

**Rationale:** The nanbeige checkpoint is 66,259,320,199 bytes across 76 files.
An accidental deep run would take hours; the metadata walk that produces that
number takes 0.6 s and is enough to refuse. The deep digest is a reproducible
identity of the tree, not a re-implementation of the model's own per-shard
admission, which stays with `nanbeige43-check`.

### Decision: The record is rooted before it is written

**Choice:** Every store path the record names (package, execution identity, and
any runtime path under `/nix/store`) gets `<gc_roots>/<profile>-<role>` before
`active.json` is replaced atomically. A missing GC-root directory that cannot be
created refuses the write.

**Rationale:** Writing the record first and rooting second leaves a window where
the record is authoritative and unrooted. The reader treats that as `broken`, but
the writer should never create it.

### Decision: `previous` is a digest, not a copy

**Choice:** `previous` carries the prior record's profile, package, timestamp,
and BLAKE3. History keeps the full records.

**Rationale:** The active record stays small and diffable, while history is the
place to look up a full prior tuple. An unreadable active record refuses the
write instead of silently discarding provenance.

## Risks / Trade-offs

- **The record pins host paths.** Receipt paths are resolved to absolute paths,
  and the checkpoint path is the host snapshot. That is what makes a later
  receipt binding unambiguous, but it means a moved checkout invalidates the
  record's evidence references. `status` does not require receipts to exist; it
  checks the roots and the package.
- **A forced record is a real record.** `--force` writes
  `admission_bypassed: true`, and `status` reports it as managed because the
  package and roots are real. The flag is refused under `TT_MODEL_UNATTENDED=1`,
  and the install check proves both.
- **GC roots are plain symlinks.** They are honoured by the collector but they
  are not created by `nix-store --add-root`, so a store path that is also a
  root's target must not be deleted by another mechanism. The reader detects a
  missing or mismatched root either way.
- **The gate-artifact path is only partly exercised.** The one gate-exit profile
  (native RWKV-7) names five gates that live in sibling derivations, so it
  records no artifacts and `check` verifies none for it. The artifact and digest
  rules are covered by fixtures; recording an artifact for a sibling would need
  a path that survives a rebuild, which a store hash does not.
- **The lock is advisory and per state root.** It serialises writers that use the
  same `TT_MODEL_STATE_ROOT`; two different roots can write independently, by
  design, because they describe different hosts or test runs.

## Non-goals

- No stop, start, reset, `lsof`, or device access; no root and no sudo. The
  privileged owner helper is step 3-4.
- No switch command. `record` selects an owner in a file; nothing acts on it.
- No change to any model's numerical gate, receipt schema, or precision preset.
