# tt-gozer lease and reconciliation design

## Source

Reviewed [`tsingletaryTT/tt-gozer`](https://github.com/tsingletaryTT/tt-gozer)
as a design reference only. It is Apache-2.0, stdlib-only Python, and provides
chip leases scoped with `TT_VISIBLE_DEVICES` plus a systemd timer that compares
lease state with the kernel's open fds; when they disagree, the kernel view wins.

This note does not copy code. No device access, reset, or timer was run.

## Current local model

`pkgs/tt-model` is a device-free registry/record/switch/run tool. Its locks are
advisory `flock` files (`.record.lock`, `.switch.lock`, `.run.lock`) and its
active record names a profile, package, runtime, mesh, and readiness identity.
`pkgs/tt-model-owner` is the only privileged program: it reports card holders
with `lsof`, stops/starts the declared unit, resets through `tt-smi`, and writes
the root-owned record.

The gaps relative to gozer are:

| Property | Local now | gozer |
| --- | --- | --- |
| Lock scope | Process-local advisory flock | Lease with holder identity and TTL |
| Crash recovery | A released fd releases the flock; no holder record to reconcile | Timer compares lease against kernel fds |
| Device visibility | Profiles pin physical IDs; no per-process `TT_VISIBLE_DEVICES` boundary | Lease-scoped visible devices |
| Authority on disagreement | Not defined | Kernel view wins |
| Stale-owner action | Manual stop/start/record | Reconciliation can flag or clear |

## Decision

Adopt the **reconciliation model**, not gozer's code or automatic ownership
clearing:

1. The owner helper's `cards` output is the kernel view: owner unit, owner pid,
   and per-card holders.
2. A lease record is advisory. It carries holder pid + process start time
   (`/proc/<pid>/stat` field 22, to defeat pid reuse), device serials, generation,
   expiry, and the last observed holder set.
3. Reconcile is read-only by default. It reports `consistent`, `stale`,
   `conflict`, or `unknown`; it never stops a unit, resets a card, or deletes a
   record. A future `--apply` requires a separate owner-authority decision.
4. If the kernel view cannot be established, the result is `unknown`, not
   `released`.

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
| Owner unit and pid | `tt-model-owner cards` | Missing/`none` with an active lease is `stale`. |
| Card holders | `tt-model-owner cards` (`lsof`) | A holder outside the owner unit is `conflict`. |
| Lease holder | Lease record | Pid start-time mismatch is `stale`; no such pid is `stale`. |
| Lease expiry | Lease record | Expired with no holders is `stale`; expired with holders is `conflict`. |
| Kernel fd view | `cards` | If it cannot be read, `unknown`. |
| DMA-BUF export state | Not observable from `/proc/*/fd` | Unsafe to infer release; `unknown`/`blocked`. |
| Reset eligibility | KMD `RESET_DEVICE` result | Only the owner helper may act; `-EBUSY` is `blocked`. |

## `TT_VISIBLE_DEVICES` evaluation

Per-lane `TT_VISIBLE_DEVICES` can keep a well-behaved child process off the
other P150, but it is not a security or ownership boundary:

- arbitrary code can unset or rewrite the variable;
- it does not stop direct `/dev/tenstorrent/*` opens;
- it is not a cgroup/namespace isolation.

It is worth adding to `tt-model run` as a **defense-in-depth hint** for a
single-card job profile, with the limitation documented. It cannot replace the
owner helper's holder check or the DMA-BUF boundary.

## Systemd timer (proposed, off by default)

A future `hardware.tenstorrent.leaseReconcile.enable` option would install:

- a oneshot service that runs `tt-model-owner cards` and a lease reader, writes
  a JSON report under `/var/lib/tenstorrent-models/reconcile/`, and exits 0 for
  `consistent`/`stale`/`conflict`/`unknown` unless `strict = true`;
- a timer with a fixed interval and `Persistent = false`;
- no stop, reset, record, or GC-root action.

The service must run as the owner helper's declared user (root) or through a
pre-authorized helper command, and must resolve `tt-smi`/`lsof`/`systemctl` by
absolute path, not PATH.

## Stale-owner test plan

Add to `pkgs/tt-model-owner/check.nix` with fake `lsof` and a temporary record:

1. **Stale no-fd**: lease active, owner pid `none`, `lsof` returns no holders,
   no export observation -> `stale`; no action.
2. **Conflict outside holder**: lease active, `lsof` returns a holder whose unit
   is not the owner -> `conflict`; no action.
3. **Positive fully released**: lease expired, no holder, owner inactive, and
   the test fixture's reset eligibility is explicitly "not attempted" ->
   `consistent`/`released`; still no action.
4. **DMA-BUF importer retained**: originating TT fd closed, importer fd retained,
   KMD reset eligibility `-EBUSY` -> `blocked`; cleanup/reassignment refused.
5. **Unknown fd view**: `lsof` failure -> `unknown`, not `released`.

Cases 1, 2, 4, and 5 must not call `systemctl stop`, `tt-smi -r`, or the record
writer. Case 3 may report `released` only when the fixture explicitly says the
export/importer state is fully released.

## Status

Design only. The timer, lease reader, and `TT_VISIBLE_DEVICES` hint are not
implemented. The current safe behavior remains: the owner helper reports holders,
the flock protects record/switch/run, and no automatic stale-owner cleanup is
performed.
