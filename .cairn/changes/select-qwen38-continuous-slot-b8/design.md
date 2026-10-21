## Decisions

### Decision: the module's closed set is {b4, b8}, and the managed drop-in names the served profile

**Choice:** express the choice twice, in the module option and in the managed drop-in
template, both bound to the same two packaged files.

**Rationale:** the module configures a host that builds the service from the package;
the managed drop-in configures the running owner on this host, where the profile
service rewrites the command. Both surfaces must be able to name the admitted
profile, and both must refuse anything else. A single file name per profile keeps the
two surfaces from drifting.

### Decision: the profile and the drop-in move together, in one operation

**Choice:** add a `switch-policy` operation to the host-local profile service instead
of swapping the profile and rewriting the drop-in in separate steps.

**Rationale:** the drop-in names a policy file inside the profile's package. A profile
swap and a drop-in rewrite in separate steps leave a window where the served command
names a file the profile does not carry. The existing `install` operation already
isolates the owner, swaps the profile, and restores the previous generation on
failure; `switch-policy` extends that discipline to the drop-in and keeps the
historical command exact. The operation needs root, which this session does not have,
so the physical deployment is performed by the owner with one command.

### Decision: the historical owner command keeps the B4 policy file

**Choice:** reconstruct the historical command with the B4 policy file, not the served
B=8 file, even though the template now names the B=8 file.

**Rationale:** the historical owner package predates the B=8 policy; naming a file it
does not carry would produce a command that cannot start. The profile service compares
that reconstruction byte for byte with the historical rollback override, so the
mapping has to name the file the historical package actually has.

### Decision: the deployment change is the served switch, not a new measurement

**Choice:** install the package the admission receipt names and verify the served
profile; do not run the admission comparison again.

**Rationale:** the receipt binds the profile to a package and an execution identity.
The deployment serves exactly that pair, so the measurement still holds. Re-measuring
would produce a second receipt for the same policy and profile.

### Decision: the module default stays at B4

**Choice:** `continuousSlotProfile` defaults to `b4`; the host's managed drop-in
selects `b8` explicitly.

**Rationale:** a module default is inherited by every host that enables the
scheduler, including hosts whose measured profile is the B4 one. The switch is a
property of this deployment, and the drop-in states it.

## Risks / Trade-offs

- The served profile changes from four to eight live slots. The admission receipt
  measured the trade on one workload: short-request median wall fell from about
  51–55 s to 3.3 s, long-request p95 rose 25.6 %, the longest pause was 5.475 s
  against a 6 s bound, and delivered throughput rose 6 %.
- The prefix cache stays enabled in the served configuration. The admission receipt
  was measured without it, so its state slots are not part of the admitted evidence.
- The rollback to the historical owner needs the original `/run` overrides, which
  only exist until the next reboot. The profile service already refuses a restore
  without them.
