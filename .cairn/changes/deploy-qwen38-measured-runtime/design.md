## Context

The service is defined by `nixos/module.nix` in this repository and launched
from a host configuration that is not in this repository. The module's
`qwen38.package` defaults to `pkgs.qwen38` (the shared 0.77 runtime) and its
service argument list carries `--decode-mode candidate` with no
batch-admission flags.

## Decisions

### Decision: Verify the target artifact before touching the host

**Choice:** Run the serving binary from the package the module default selects,
on the isolated devices, and record readiness plus one greedy request before any
host change.

**Rationale:** The switch restarts production. A readiness gate on the exact
artifact removes the "does the new package even start" risk from the moment the
service goes down.

### Decision: Keep batch serving off

**Choice:** Do not pass `--batch-admission` in the service.

**Rationale:** The admitted B=8 receipt states that it does not admit arbitrary
continuous batching. Serving the batch coordinator on the strength of a
width-8 measurement receipt would exceed that evidence. A batched serving route
needs its own admission, which is separate work.

### Decision: Do not regenerate the artifact manifest

**Choice:** Repair only the two-token merge damage.

**Rationale:** Commit `680afa7` recorded the manifest's entry drift as a known
pre-existing state and deliberately did not regenerate it. Regenerating now
would convert a red gate into a green one and hide drift the owner chose to
keep visible. The repair restores the gate's ability to report that drift.

### Decision: The host edit is recorded, not invented

**Choice:** Name the exact host edit and the values it must select, but do not
guess at a configuration file that this repository does not contain.

**Rationale:** Writing a host configuration from assumption would produce a
switch that cannot be reviewed. The blocker is the location of that file, and it
is recorded rather than papered over.

## Risks / Trade-offs

- The switch changes served behavior on a candidate decode mode; the readiness
  gate and the post-switch request check bound that risk but cannot prove
  quality beyond the existing single-stream gate.
- If the host configuration cannot be located, the change stays active with its
  verification tasks unresolved; that is the honest state, not a completed one.
