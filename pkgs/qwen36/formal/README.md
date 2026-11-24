# Qwen3.8 Hybrid Prefix-Cache Lifecycle TLA+ Model

Bounded TLA+ specification (`PrefixCacheLifecycle.tla`) and TLC check suite for the
Qwen3.8 continuous-slot hybrid prefix cache and slot cancellation lifecycle, resolving
`OnixResearch/tt-onix#39`.

## Pinned Source Commit

All line references below are pinned to commit
`5fd2ad6f339007560a678bcb5fa3126b84e95b26` (identical service core to `5ed5e90cdd265443fa2f9b73e4ce46aeaa71837a`):

- `pkgs/qwen36/service/qwen36_service/continuous_slot_core.py`
- `pkgs/qwen36/service/qwen36_service/prefix_cache_core.py`
- `pkgs/qwen36/service/qwen36_service/continuous_slot_service.py`

## Action-to-Source Mapping

| TLA+ Action | Source Mapping (`file:line@commit`) | Modeled Transition |
| --- | --- | --- |
| `AdmitRequest` | `continuous_slot_core.py:472-653, 727-814@5fd2ad6f339007560a678bcb5fa3126b84e95b26` and `prefix_cache_core.py:239-267@5fd2ad6f339007560a678bcb5fa3126b84e95b26` | Lowest free slot selection (`free[0]`), `lookup`, `_free_run`, `eviction_order` (leaf-before-root, `pins == 0` guard), `_repin(+1)`, and `_plan_donation` (new root, new leaf, or in-place supersede of a single-pinned non-root hit reusing `state_index` and transferring its pin) |
| `ObservePrefill` | `continuous_slot_core.py:868-885@5fd2ad6f339007560a678bcb5fa3126b84e95b26` | Prefill completion (`prefilling -> active` or `complete`), transitioning the planned `donation_entry` to `ready = TRUE` and making its donated KV prefix blocks immutable read-only |
| `FinishDecode` | `continuous_slot_core.py:888-894@5fd2ad6f339007560a678bcb5fa3126b84e95b26` | Active decode slot completes with `stop` or `length` |
| `CancelSlot` | `continuous_slot_core.py:427-451@5fd2ad6f339007560a678bcb5fa3126b84e95b26` and `continuous_slot_service.py:327-338@5fd2ad6f339007560a678bcb5fa3126b84e95b26` | Cancels a request in `RUNNING_STATES` (`prefilling` / `active`) to `complete(cancelled)` while leaving already-complete slots unchanged |
| `BoundaryRetire` | `continuous_slot_core.py:679-697@5fd2ad6f339007560a678bcb5fa3126b84e95b26` | Retires every `SLOT_COMPLETE` slot at the decode-window boundary, freeing its KV run and calling `_repin(prefix_entry, -1)` |
| `InvalidatePrefixCache` | `continuous_slot_core.py:980-986@5fd2ad6f339007560a678bcb5fa3126b84e95b26` | Conservative invalidation when KV writes cannot be tracked or traces are released: drops `pins == 0` entries and retains pinned entries |
| `StructuredRead` | `continuous_slot_service.py:433, 469-491@5fd2ad6f339007560a678bcb5fa3126b84e95b26` and `continuous_slot_core.py:989-1008@5fd2ad6f339007560a678bcb5fa3126b84e95b26` | Nondeterministic structured-read write set guarded by `not self._occupied()` and `protected_blocks = cached_blocks(prefix_entries)`, followed by `invalidate_written_prefixes` dropping every intersecting entry (pinned or unpinned) |
| `DeviceFailure` | `continuous_slot_core.py:940-963@5fd2ad6f339007560a678bcb5fa3126b84e95b26` | `quarantine_group`: quarantines occupied slots (`FINISH_FAILED`) and drops every cached entry |
| `ReleaseQuarantine` | `continuous_slot_core.py:966-977@5fd2ad6f339007560a678bcb5fa3126b84e95b26` | Frees quarantined slots after device reset |

## Bounds and Checked Invariants

Checked bounds: `NumSlots = 2`, `MaxEntries = 3`, `NumBlocks = 8`, `NumStateSlots = 2`, `MaxCachedBlocks = 6`.

1. `LiveBlocksDisjointFromOtherRuns`: No two live slots overlap on `ownBlocks`, and no live slot's readable blocks (`prefixBlocks` + immutable post-prefill `donatedBlocks`) or writable blocks intersect another live slot's writable blocks.
2. `ReadyEntryBlocksImmutable`: A ready entry's blocks are never written after it becomes ready (neither by a new slot's own run nor by a structured read).
3. `UniqueCarriedStateSlots`: No two entries in `entries` share a `stateIndex`.
4. `PinAccountingSound`: Pins never go negative, always equal the exact count of live non-quarantined slots referencing each entry, and unpinning an entry absent from the table never occurs on protected paths.
5. `AtMostOneRoot`: At most one `root = TRUE` entry exists in `entries`.
6. `CancelPreservesFinishReason` & `CancelledReleasedAtBoundary`: Cancelling never overwrites an already-finished slot's `finish_reason`, and every complete/cancelled slot is retired at the next boundary.

## Resolution of the `invalidate_written_prefixes` vs `design.md` Mismatch

The model and source confirm why `invalidate_written_prefixes` (`continuous_slot_core.py:989-1008`)
drops **all** entries whose block ranges intersect `written_blocks` (pinned or unpinned),
while `invalidate_prefix_cache` (`continuous_slot_core.py:980-986`) retains pinned entries:

- If a structured read physically writes a block belonging to a cached entry, those KV bytes
  are overwritten and cannot safely remain in the table (already enforced by `pkgs/qwen36/check.py:4017-4097`).
- Live-reader and immutability safety hold on the service path because `continuous_slot_service.py`
  enforces **two guards**: (1) reads execute only when `not self._occupied()` (line 433, so `pins == 0`
  for every entry), and (2) `_serve_reads` passes `protected_blocks = cached_blocks(prefix_entries)`
  (lines 471–472) so reads do not write cached blocks.
- Removing `protected_blocks` (`MutantNoProtectedBlocksGuard.cfg`) violates `ReadyEntryBlocksImmutable`
  even on an idle device; removing both `protected_blocks` and idle-drain (`MutantPinnedDropOnRead.cfg`)
  allows `invalidate_written_prefixes` to overwrite and drop a pinned entry while a live slot reads it.
- `.cairn/changes/qwen38-hybrid-prefix-cache/design.md` (line 49) has been updated to match this exact
  contract so code, tests, model, and design agree.

An additional modeling check confirmed why `next_entry_id` monotonicity (`continuous_slot_core.py:780`)
is essential: if a freshly donated entry could reuse an ID still held in a prefilling slot's
`donation_entry` after `invalidate_prefix_cache`, `observe_prefill` could prematurely mark another
slot's unready donation `ready = TRUE`.

## Checker Selection and Verified TLC Results

Checker: `tlc` from `pkgs.tlaplus` (TLC2 2.19). `everystate` is not packaged in the repository's
pinned `nixpkgs`, so TLC is used directly in the `qwen38` / `qwen36` Nix check via
`pkgs/qwen36/prefix-cache-tla-check.py`.

| Configuration | Guard Removed | Expected | TLC Verdict | States Generated | Distinct States | Violated Invariant | Trace Length |
| --- | --- | --- | --- | ---: | ---: | --- | ---: |
| `PrefixCacheLifecycle.cfg` | None (all guards enabled) | PASS | PASS | 23,848,756 | 3,377,457 | — | 0 |
| `MutantNoUnpinnedEvictionGuard.cfg` | `GuardEvictUnpinnedOnly = FALSE` (`pins == 0` in `eviction_order`) | FAIL | FAIL | 9,330 | 2,625 | `LiveBlocksDisjointFromOtherRuns` | 6 |
| `MutantNoProtectedBlocksGuard.cfg` | `GuardProtectedBlocks = FALSE` (`protected_blocks` in `_serve_reads`) | FAIL | FAIL | 1,542 | 491 | `ReadyEntryBlocksImmutable` | 7 |
| `MutantPinnedDropOnRead.cfg` | `GuardProtectedBlocks = FALSE`, `GuardReadRequiresIdle = FALSE` | FAIL | FAIL | 927 | 114 | `ReadyEntryBlocksImmutable` | 5 |
| `MutantCancelCompletedSlot.cfg` | `GuardCancelRunningOnly = FALSE` (`RUNNING_STATES` in `cancel`) | FAIL | FAIL | 135 | 46 | `CancelPreservesFinishReason` | 5 |
| `MutantSkipBoundaryRetire.cfg` | `GuardRetireCompleteAtBoundary = FALSE` (`SLOT_COMPLETE` retire in `plan_boundary`) | FAIL | FAIL | 84 | 42 | `CancelledReleasedAtBoundary` | 5 |

*(Mutant state counts vary slightly with TLC multi-worker scheduling; invariant violation and pass/fail verdicts are deterministic.)*
