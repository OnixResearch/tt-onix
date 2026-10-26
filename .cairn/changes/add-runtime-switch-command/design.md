## Context

A serve profile is the only thing that can own the cards across requests, and the
only way to make it the owner is to stop whatever owns them, reset the boards,
start the service, and prove it is the service the record names. The manual
procedure exists in prose (`docs/qwen38-served-deployment.md`), and it has no
rollback: a service that starts with the wrong package, or with a card still held
by another process, is discovered by a client, not by the operator.

The switch is the one command that changes the machine. It stays device-free in
`tt_model.py`; every privileged step goes through the owner helper, and the
`/healthz` and containment traffic is loopback only.

Nothing in this change was applied to the host. The example module is a fragment
for review; the record on this host is untouched; no unit, rule, or device was
changed.

## Decisions

### Decision: The switch writes the candidate record before it starts

**Choice:** After the stop, reset, re-check, and containment, the switch commits
the candidate record and its GC roots, then starts the owner, waits for
readiness, and reads identity back. The final commit rewrites the record with the
observed identity and adds a `.verified` history entry.

**Rationale:** If the process dies between the start and the record write, the
machine has an owner nobody recorded. Writing first means every live owner has a
record, and the rollback is what protects the window.

### Decision: Rollback restores bytes, not a recomputed record

**Choice:** The previous `active.json` is saved before the commit and restored
byte for byte. A previous owner that was running is started again and must pass
its own readiness and identity checks. If it cannot, the restored record is moved
to history, `active.json` is removed, and `switch-failure.json` records why.

**Rationale:** A recomputed "previous" record could differ from the one that was
verified. Restoring bytes makes `cmp` the test, which the install check does.
And an owner that cannot be re-verified must not stay recorded as active.

### Decision: The candidate is checked again after the reset

**Choice:** `check` runs before the stop and again after the reset.

**Rationale:** A reset is when the store or a policy path could have moved under
the switch (a GC, a concurrent rebuild). Re-checking costs one validation pass and
turns a silent wrong-package start into a refusal before anything is written.

### Decision: Containment is a declared command, not special-cased Python

**Choice:** A profile may declare a repository-relative containment command, its
arguments (only `@package@` and `@package_identity@` expand), and the JSON verdict
it must print. qwen38 declares the checker it already ships.

**Rationale:** The model package owns its containment rule. Copying the checker
into `tt-model` would freeze one model's admission logic into the shared tool, and
`--force` must never be able to skip it.

### Decision: Identity comes from the instance, with a declared source

**Choice:** `readiness.identity_source` is `healthz` by default and
`decode-evidence` when declared (qwen38). The record pins model identity, package
identity, device ids, and `simulated`; runtime identity and profile id are
required to be present and are recorded. A re-switch compares runtime identity
against the recorded one.

**Rationale:** The served `/healthz` payload does not carry package, runtime, or
profile identity today, and changing it would move the qwen38 derivation, which
this step must not do. The decode evidence does carry them and is what
`qwen38-deploy` already reads. The two fields the profile cannot predict are
recorded rather than compared on the first switch, then compared on every later
one — which is the strictest check the current service supports.

### Decision: The host fragment is an example plus an evaluation check

**Choice:** The activation lives in `nixos/examples/qwen38-tt-model-switch.nix`,
exported as a module but imported by nothing. `checks/tt-model-host-config.nix`
proves it adds exactly `qwen38-p150x2`, renders one NOPASSWD rule for the helper,
keeps the reviewed serving options, and rejects a wider grant.

**Rationale:** The host configuration is not in this repository, and applying it
is the owner's decision. An example that is proved to evaluate is reviewable; an
example that is only read is not.

## Risks / Trade-offs

- **The record window is real.** Between the candidate commit and the verdict,
  `active.json` names the new package but the service may not be ready. A crash
  there leaves an unverified record; the next `status` reports it as unmanaged
  only if its roots or package moved, so the window depends on the switch being
  the thing that resumes. Rollback covers every handled failure, not a SIGKILL.
- **Identity is compared where the record can predict it.** Runtime identity and
  profile id are recorded, not predicted, on a first switch. Extending `/healthz`
  to carry them would let the record pin them, at the cost of a qwen38 rebuild and
  a fresh admission.
- **A foreign holder blocks the switch.** `cards` reports unit `-` for any process
  whose cgroup is unreadable, so an unusual setup refuses rather than guessing.
- **The reset is unconditional for TT-Metal serve profiles.** That matches the
  documented rule, but it costs a board reset on every switch even when the owner
  was already stopped.
- **The example names host paths and units.** It is a host fragment, not portable
  configuration; the eval check only proves it renders.

## Non-goals

- No host activation and no `nixos-rebuild`.
- No change to the qwen38 package, its unit definition, or any existing receipt.
- No change to the helper's subcommand set; the switch uses `cards`,
  `stop-owner`, `start-owner`, and `reset` as they are.
