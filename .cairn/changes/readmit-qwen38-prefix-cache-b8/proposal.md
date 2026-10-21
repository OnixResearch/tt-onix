## Why

The deployed configuration pairs two features: the eight-slot continuous-slot scheduler
and the hybrid prefix cache. Their receipts were earned separately and neither covers
the pair.

- The prefix cache was admitted on 2026-09-27 (task p7) for package
  `vvi5i8wy…-qwen38-0.1.0`, with the four-slot scheduler profile, and without the
  eight-slot profile in existence.
- The eight-slot scheduler was admitted on 2026-10-21 for package
  `inkfvsjq…-qwen38-0.1.0`, with the prefix cache's state slots absent from the
  comparison.

So the configuration that now serves traffic has no receipt. Two things could differ
in the pair: the cache's saved state slots sit beside a wider resident trace, and the
cache's hit and miss costs are measured against a scheduler that refills eight slots
instead of four. A prefix-cache root also does not record which scheduler profile it
served, so a receipt could be read as covering a scheduler the roots never ran.

## What Changes

- Add a schema-5 prefix-cache admission policy that names the continuous-slot profile
  every root must have served, and keeps every schema-2 gate bound.
- Make the admission core bind the scheduler profile when the policy requires it, and
  leave the older schemas validating the committed roots unchanged.
- Record the served scheduler profile in each root's identity, and require the
  listener to report one.
- Export the schema-5 policy from the package and bind it into the execution identity.
- Run the ABBA comparison on the current package with the eight-slot profile at the
  served limits, and re-run the B=8 scheduler admission on the same package so the
  served package carries both receipts.
- Move the served profile to that package.

## Impact

- **Files**: `pkgs/qwen36/prefix-cache-admission-contracts.ncl`, the new
  `pkgs/qwen36/prefix-cache-admission-policy-v5.ncl`,
  `pkgs/qwen36/service/qwen36_service/prefix_cache_admission_core.py`,
  `scripts/qwen38-prefix-cache-screen.py`, `pkgs/qwen36/default.nix`,
  `pkgs/qwen36/check.py`, the adoption document, this change directory, and its
  evidence.
- **Testing**: the Qwen package check with a new device-free probe, the repository
  quality gate, and one physical ABBA campaign with four fresh processes per workload.
- **Risk**: none to the served behaviour. The new schema only adds a field and a
  gate; the older schemas and their committed roots keep validating as they did.
