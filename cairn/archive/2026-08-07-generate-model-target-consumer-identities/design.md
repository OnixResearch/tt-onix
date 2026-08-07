## Context

`model-targets/profiles.ncl` exports canonical JSON and the freshness check compares it with checked-in generated data. Nix target construction uses that data. Several downstream programs still embed the same facts as source constants.

The repeated facts include model ID, revision, dimensions, artifact hashes, topology, state shape, precision, tolerance values, and evidence adapter identity. These values cross language boundaries and can drift independently.

## Decisions

### Decision: Nickel owns consumer identity data

**Choice:** Extend the typed profile with every value that must remain identical across runtime and evidence consumers. Keep computational equations and control flow outside the generated projection.

**Rationale:** Identity data belongs in one reviewed authority. Generated algorithms would hide behavior and make code review harder.

### Decision: Generate data modules, not source logic

**Choice:** Generate deterministic Rust, Python, C++, and Nix data modules or include files. Each output contains constants, typed records, and the canonical profile BLAKE3 identity only.

**Rationale:** Data generation removes duplication while preserving readable functional cores.

### Decision: One hash grammar applies everywhere

**Choice:** Accept only exact lowercase hexadecimal text for BLAKE3 and required artifact hashes. Generated consumers use shared length constants and reject uppercase, mixed-case, short, long, or non-hex values.

**Rationale:** Equal identities need equal text representation. Case-normalizing one boundary can hide malformed input.

### Decision: Freshness is a required flake check

**Choice:** Regenerate every consumer projection in a temporary directory and compare it byte for byte with checked-in files. Report the profile and output that differs.

**Rationale:** Checked-in generated files support Nix builds, but only a freshness gate keeps them trustworthy.

### Decision: Runtime and evidence bind the same profile identity

**Choice:** Every runtime, command, and evidence receipt exposes the canonical profile identity. Package construction fails when a consumer uses a different generated identity.

**Rationale:** Matching source constants are not enough. The built outputs need an explicit common identity.

### Decision: Existing model interfaces remain stable

**Choice:** Preserve package attributes, command names, receipt kinds, model revision, topology, and current accepted numerical boundaries.

**Rationale:** This change removes identity duplication. It does not redefine runtime behavior.

## Validation Strategy

Positive fixtures will compare all generated projections with one canonical profile. Package tests will verify that runtime, command, evidence, and report outputs expose the same identity.

Negative fixtures will cover stale generated files, missing consumer fields, uppercase and mixed-case hashes, wrong dimensions, wrong topology, changed tolerance data, and a consumer built from another profile.

Existing RWKV device-free checks and the Qwen candidate change must keep their current behavior.

## Risks / Trade-offs

- Generated files increase review noise when a profile changes.
- A broad first conversion can touch many packages. Conversion will proceed by identity groups with freshness checks after each group.
- C++ and Python include formats need stable formatting rules.
- A profile field that controls behavior needs explicit review before it enters generated data.
