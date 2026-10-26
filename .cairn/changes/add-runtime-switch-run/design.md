## Context

Step 2 gave the registry a checker and a record writer. Both are device-free.
Taking the cards needs operations that are not: stopping a unit, asking which
process holds a card, resetting the boards, and writing a root-owned record. The
quality policy forbids those strings in the device-free gate, and step 1's
install check asserts that `tt_model.py` does not contain them, so the privileged
work has to live somewhere else.

The boundary chosen here is one small program with five subcommands. Everything
it can do is visible in one file, it prints what it is about to run, and the
NixOS module grants it exactly one sudo command.

No hardware was touched while implementing this. The live run is deferred: the
sandbox exercises the sequence against fake helpers, and the host was used only
for refusals that stop before any privileged command runs.

## Decisions

### Decision: The helper is separate, and both directions are asserted

**Choice:** `tt-model-owner` owns `/dev/tenstorrent`, `tt-smi`, the unit name,
and `systemctl`. `tt_model.py` names none of the device or reset strings, and the
helper's check asserts that it does name them.

**Rationale:** The policy string list is the contract. Asserting only one
direction would let the boundary rot: the device-free tool could quietly start
touching devices, or the helper could lose the only code that knows how to. Both
checks now fail if either happens.

### Decision: Every override is a test-only capability

**Choice:** `TT_MODEL_STATE_ROOT`, `TT_MODEL_GC_ROOTS`, `TT_MODEL_OWNER_UNIT`,
and the three command paths are refused unless `TT_MODEL_OWNER_TEST=1`.

**Rationale:** A privileged helper that follows an environment variable to a
different state root is one shell line away from writing a record where nothing
reads it, or stopping a unit someone else named. The test flag makes the intent
explicit and keeps the sandbox able to exercise everything.

### Decision: The helper proves the host recorded it

**Choice:** The helper reads `<state-root>/owner-helper` and refuses unless it
resolves to its own store path. The NixOS module writes that pointer.

**Rationale:** A rebuilt helper with a new store path must not inherit authority
from the one the host approved. Step 2's record has no helper field, so the
pointer is the host's statement instead. It is created by the same module that
grants the sudo rule, which keeps the two facts together.

### Decision: `cards` reports facts, `run` decides

**Choice:** The helper reports the owner unit, the owner PID, and every holder as
`<device> <pid> <unit> <command>`. `tt-model run` refuses when any holder's unit
differs from the owner unit.

**Rationale:** The helper resolves holder units from `/proc/<pid>/cgroup`, which
is exact and readable without privilege; the policy decision stays in the
device-free tool where it can be tested without a helper.

### Decision: Reset follows the profile and the stopped owner

**Choice:** `run` resets when the profile's `reset.on_release` is not `never`,
or when it actually stopped an owner.

**Rationale:** The documented rule is that a reset is required after a TT-Metal
owner stops and before native work. A native job profile (`on_release = never`)
started after a stopped owner still needs that reset; a TT-Metal job profile is
reset unconditionally because a known board state is cheaper than a diagnosis.

### Decision: The job gets the profile environment and nothing else

**Choice:** The child environment is the profile's `env` plus `PATH`. No caller
variable is inherited, and the job's own output is redirected to the runner's
stderr so the runner's report stays parseable.

**Rationale:** A caller with `MESH_DEVICE` or `TT_CACHE_PATH` exported would
otherwise change what the job does without any record of it. The install check
proves this by exporting a decoy `MESH_DEVICE` and reading the job's own output.

### Decision: A run never becomes the owner

**Choice:** `run` writes `last-run.json` and a `runs/` event, never
`active.json`; `status` reports the last run beside the active record and lets it
affect no verdict.

**Rationale:** A finished job holds nothing. Treating its receipt as an active
owner would make `status` claim the cards are in use when they are free.

### Decision: One sudo rule, asserted at evaluation time

**Choice:** The module renders one rule for one user with one command and
`NOPASSWD`, plus three assertions: the operator is not `ALL`, exactly one rule
names the helper binary and grants no other command, and no rule for the operator
grants anything inside the helper package that is not that binary. The check also
proves the `qwen38-p150x2` unit is byte-for-byte the same projection with and
without the option.

**Rationale:** A grant is a security boundary, so it is proved where it is
rendered rather than reviewed by eye. The negative evaluations are part of the
check, not a comment.

## Risks / Trade-offs

- **The pointer is a second artifact to keep current.** If a host installs a new
  helper without updating `/var/lib/tenstorrent-models/owner-helper`, the helper
  refuses. That is the intended failure: silence would mean an old binary acting
  with new authority.
- **`cards` depends on `/proc/<pid>/cgroup`.** A process in a different cgroup
  hierarchy, or one whose cgroup is unreadable, is reported with unit `-`, which
  `run` treats as foreign. Fail-closed, at the cost of a false refusal in exotic
  setups.
- **The run lock is separate from the record lock.** A `run` and a `record` can
  proceed concurrently against the same state root. They write different files
  and neither writes the other's, so this is safe today; a future switch that
  does both must take both locks.
- **The helper's `record` subcommand delegates to `tt-model record`.** That keeps
  one writer, but it means the privileged path inherits the unprivileged writer's
  behaviour. The helper pins the state root and GC-root directory, so the
  delegation cannot be redirected.
- **The live sequence is unexercised.** Stop/start/reset against a real unit and
  real boards is deferred; only the refusal paths and the fake-helper sequence
  ran here.

## Non-goals

- No switch command. `run` is for job profiles; `switch` for serve profiles is
  the next step, together with the readiness and identity readback it needs.
- No enabling or unmasking of any unit. The helper's `start-owner` refuses a
  masked unit and says so.
- No change to any model's numerical gate, receipt schema, precision preset, or
  existing receipt.
