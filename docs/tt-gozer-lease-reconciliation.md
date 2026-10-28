# tt-gozer lease and reconciliation design

## Source

Reviewed [`tsingletaryTT/tt-gozer`](https://github.com/tsingletaryTT/tt-gozer)
as a design reference only. It is Apache-2.0, stdlib-only Python, and provides
chip leases scoped with `TT_VISIBLE_DEVICES` plus a systemd timer that compares
lease state with the kernel's open fds; when they disagree, the kernel view wins.

This note does not copy code. No device access, reset, or timer was run.

## Current local model

`pkgs/tt-model` is a device-free registry/record/switch/run tool with a
read-only reconcile reader. Its locks are advisory `flock` files
(`.record.lock`, `.switch.lock`, `.run.lock`) and its active record names a
profile, package, runtime, mesh, and readiness identity.
`pkgs/tt-model-owner` is the only privileged program: it reports card holders
with `lsof`, stops/starts the declared unit, resets through `tt-smi`, and writes
the root-owned record.

The gaps relative to gozer are:

| Property | Local now | gozer |
| --- | --- | --- |
| Lock scope | Process-local advisory flock plus a read-only reconcile reader | Lease with holder identity and TTL |
| Crash recovery | A released fd releases the flock; reconcile flags a stale record without clearing it | Timer compares lease against kernel fds |
| Device visibility | Single-card job profiles receive a `TT_VISIBLE_DEVICES` hint | Lease-scoped visible devices |
| Authority on disagreement | Kernel view wins in `tt-model reconcile` (flag only) | Kernel view wins |
| Stale-owner action | Manual stop/start/record | Reconciliation can flag or clear |

## Decision

Adopt the **reconciliation model**, not gozer's code or automatic ownership
clearing:

1. The owner helper's `cards` output is the kernel view: owner unit, owner pid,
   and per-card holders.
2. A lease record is advisory. The active record may carry a `lease` object with
   holder pid + process start time (`/proc/<pid>/stat` field 22, to defeat pid
   reuse), physical device ids, generation, expiry, and the last observed holder
   set. The block is optional: a record written without it is reconciled from
   the cards view and reports `expired: null` rather than guessing an expiry.
   The writer does not yet populate lease fields; the reader accepts them so a
   later writer can add them without a reader migration.
3. Reconcile is read-only. It reports `consistent`, `stale`, `conflict`,
   `unknown`, or `no-record`; it never stops a unit, resets a card, or deletes a
   record. A future `--apply` requires a separate owner-authority decision.
4. If the kernel view cannot be established, the result is `unknown`, not
   `released`.

The read-only command is `tt-model reconcile --cards PATH [--json]`. `PATH` is a
captured `tt-model-owner cards` file; reconcile parses it and never executes the
helper, opens a card node, or queries a device. It also does not need the
runtime-profile registry. The report always carries `dma_buf: "unobserved"`,
`release_claimed: false`, `reset: "not-attempted"`, and `action: "none"`.
`reconcile` exits 0 for every observation, including `stale`, `conflict`, and
`unknown`; a nonzero status is reserved for a usage or parser error. A timer
being healthy therefore means "the observation completed", not "the cards are
free".

## DMA-BUF lifetime boundary

The pinned KMD (`ttkmd-2.10.0`) `TENSTORRENT_IOCTL_EXPORT_TLB_DMABUF` contract
retains an exported TLB window while a DMA-BUF importer holds it, even after
`FREE_TLB` or closing the originating `/dev/tenstorrent` fd. `RESET_DEVICE`
returns `-EBUSY` while exports are live.

Therefore:

- **no visible `/dev/tenstorrent/*` fd is not proof of quiescence**;
- an active export/importer is `unknown` or `blocked`, never `released`;
- reconcile must not weaken KMD's reset refusal or add automatic reset;
- any cleanup waits for both the fd view and KMD reset acceptance.

This is a design caveat from the pinned driver, not a reproduced failure.

## Reconciliation inputs

| Input | Source | Rule |
| --- | --- | --- |
| Owner unit and pid | `tt-model-owner cards` | Missing/`none` with no holders is `stale`; with a holder still visible, `conflict`. |
| Card holders | `tt-model-owner cards` (`lsof`) | A holder outside the owner unit is `conflict`; an unattributable (`-`) holder is `unknown`. |
| Lease holder | Lease record | Pid start-time mismatch is `stale`; no such pid is `stale`. An unreadable `/proc/<pid>/stat` is `unknown`. |
| Lease expiry | Lease record | Expired with no visible holder and no owner pid is `consistent` (observational agreement only). Expired while the owner still runs is `stale`; expired with a visible holder is `conflict`. |
| Kernel fd view | `cards` | If it cannot be read, `unknown`. |
| DMA-BUF export state | Not observable from `/proc/*/fd` | Unsafe to infer release; `unknown`/`blocked`. |
| Reset eligibility | KMD `RESET_DEVICE` result | Only the owner helper may act; `-EBUSY` is `blocked`. |

### Implemented precedence

`tt-model reconcile` applies these checks in order and stops at the first
decisive one:

| Condition | State |
| --- | --- |
| No `active.json` | `no-record` |
| Record unreadable, wrong schema, or a malformed `lease` block | `unknown` |
| Cards file missing/unreadable, no owner unit, or no card line | `unknown` |
| Cards view does not list exactly the cards the record's mesh names | `unknown` |
| A holder whose unit is known and is not the owner unit | `conflict` |
| A holder whose unit is `-` (unattributable) | `unknown` |
| Lease holder pid gone or its `/proc/<pid>/stat` start time changed | `stale` |
| Lease expired, holders visible | `conflict` |
| Lease expired, owner still running, no holders | `stale` |
| Lease expired, no owner pid, no holders | `consistent` |
| Owner pid `none` and no holders | `stale` |
| Owner pid `none` but a holder remains | `conflict` |
| Owner pid present but no holder is visible | `unknown` |
| Owner pid present and every holder is in the owner unit | `consistent` |

A partial cards view is never treated as evidence: if the record names devices
`0,1` and the capture covers only `0`, the result is `unknown`. A holder in an
unattributable cgroup is also `unknown`, which is how a retained DMA-BUF
importer surfaces without `/proc` being able to show the export itself.

## `TT_VISIBLE_DEVICES` evaluation

Per-lane `TT_VISIBLE_DEVICES` can keep a well-behaved child process off the
other P150, but it is not a security or ownership boundary:

- arbitrary code can unset or rewrite the variable;
- it does not stop direct `/dev/tenstorrent/*` opens;
- it is not a cgroup/namespace isolation.

**Implemented as a defense-in-depth hint.** `tt-model run` now sets
`TT_VISIBLE_DEVICES` from the job profile's single `mesh.physical_device_ids`
entry (for example `"0"` for the single-card RWKV profile). Multi-card profiles
get no value, because naming both cards adds nothing. The value is computed from
the profile, not accepted from the caller: a caller's `TT_VISIBLE_DEVICES` does
not reach the job, and the run event records the assigned hint in
`tt_visible_devices`.

This narrows the devices a runtime that honors the variable will enumerate. It
cannot replace the owner helper's holder check or the DMA-BUF boundary, and it
is not a claim that the other card is unreachable by arbitrary code.

## Systemd timer (implemented, off by default)

`hardware.tenstorrent.ttModel.leaseReconcile.enable` is off by default and
installs:

- one oneshot service, `tt-model-lease-reconcile.service`, which runs the owner
  helper's read-only `cards` command with an absolute store path and pipes the
  capture into `tt-model reconcile --cards ... --json` with an absolute store
  path. The JSON report goes to the journal; no report file is written. There is
  no stop, start, reset, record, switch, run, or GC-root action.
- one timer, `tt-model-lease-reconcile.timer`, with `OnBootSec` and
  `OnUnitActiveSec` set to the `leaseReconcile.interval` option (default
  `5min`), `Persistent = false`, `Unit = "tt-model-lease-reconcile.service"`,
  and `wantedBy = [ "timers.target" ]` only.
- the service reads the owner helper's fixed state root
  (`/var/lib/tenstorrent-models`) through `TT_MODEL_STATE_ROOT` and runs as root,
  which is the identity `lsof` needs for a complete card-holder view. Its `PATH`
  is a minimal Nix store path; both invoked programs are absolute.

Enabling the timer requires `hardware.tenstorrent.ttModel.enable`, so the
`owner-helper` pointer the helper validates is present. The timer is read-only:
it may report `stale` or `conflict`, but clearing or acting on either remains a
separate, manual owner decision.

## Stale-owner test matrix

The mocked cases live in the `tt-model` install check, with an additional
stale-lock run in `checks/tt-model-host-config.nix`:

1. **Stale no-fd**: lease active, owner pid `none`, cards view has no holders ->
   `stale`; no action.
2. **Conflict outside holder**: a holder whose unit is not the owner ->
   `conflict`; no action.
3. **Positive fully released**: lease expired, no holder, owner inactive ->
   `consistent`; `dma_buf` stays `unobserved` and `release_claimed` stays false.
4. **DMA-BUF importer retained**: an unattributable holder (unit `-`) with an
   expired lease -> `unknown`; release and reset are refused.
5. **Unknown fd view**: a missing/unreadable cards capture -> `unknown`, never
   `released`.
6. **Stale-lock**: a lease holder pid that no longer exists -> `stale`; the
   record is flagged, not cleared.

Every case asserts `dma_buf: "unobserved"`, `release_claimed: false`,
`reset: "not-attempted"`, and `action: "none"`. No case executes the owner
helper, opens a device, or requires a real card.

## Status

Implemented in `pkgs/tt-model` and `nixos/module.nix`. The command is
`tt-model reconcile --cards PATH [--json]`; it is read-only, does not load the
runtime-profile registry, and never executes the owner helper. The NixOS timer
is off by default and only runs that command. Flock continues to protect
record/switch/run; reconciliation never clears a stale owner automatically, and
the DMA-BUF export state remains unobservable, so no case claims a release.
