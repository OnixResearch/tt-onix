# Step-kernel hazard audit: the staged scalar reads (V4)

Recorded 2026-10-22 as a **source audit** of commit `ed2f4a1` ("qwen38: stage the step
reader's scalar reads behind one NOC barrier") over the GDN decode step kernels carried in
`pkgs/qwen36/gdn-decode-step.patch`. This is a reading of the changed sources against the
pinned ISA and tt-metal sources below. No silicon was executed for it, no simulator was run,
and nothing here is a timing or admission claim.

## Method

The patch is a patch-of-patches applied with `patch -p1` over the tt-metal tree
(`pkgs/qwen36/default.nix:222` `patches = executionPatches;`). Both the parent
(`ed2f4a1^`) and the candidate (`ed2f4a1`) versions of the patch were unpacked into throwaway
trees (`/tmp/v4audit/par`, `/tmp/v4audit/cur`) by taking the added lines of each new-file
hunk, so the audited files are the real kernel sources and no tracked file was modified.
Line numbers below are file-relative and equal the patch's added-line offsets; the reader
hunk starts at `gdn-decode-step.patch:1000` with `@@ -0,0 +1,175 @@`, so reader line *N* is
patch line *1003 + N*.

All seven kernel files were extracted and compared. **Exactly one file differs**:

```
cur/models/demos/blackhole/qwen36/tt/gdn/kernels/{conv_compute,conv_reader,conv_writer}.cpp  identical
step_compute.cpp 369 lines  identical
step_reader.cpp  175 lines  DIFFERS (parent 166 lines)
step_writer.cpp   64 lines  identical
decode_step.py, decode_fusion_candidate.py, tp.py  unchanged in this commit
```

So the audit scope is `step_reader.cpp` (changed) plus the compute-kernel invariants the
change leans on (`step_compute.cpp`, `step_writer.cpp`, and the host-side CB geometry in
`decode_step.py`).

Audit classes were taken from the repository skills in `.pi/skills/ttllk-*`
(`noc-sync`, `dataflow-cb-sync`, `srcreg-bank-sync`, `mmio-race`, `cfg-word-overlap`,
`reconfig-stall`, `semaphore-handshake`, `mailbox-sync`, `instruction-latency`), bounded to
the classes a dataflow reader kernel can contain. Sources consulted, in the ladder order of
`ttllk-race-audit-all`:

| Source | Role | Reachable |
| --- | --- | --- |
| `.pi/references/tenstorrent/tt-isa-documentation/BlackholeA0/TensixTile/BabyRISCV/MemoryOrdering.md` | BH RISC load/store, L0 cache, `fence` semantics | yes |
| `.pi/references/tenstorrent/tt-isa-documentation/BlackholeA0/NoC/` (`Atomics.md`, `Counters.md`, `MemoryMap.md`, `RoutingPaths.md`) | BH NoC | yes — **no `Ordering.md` exists for BlackholeA0** (only `WormholeB0/NoC/Ordering.md`); see the coverage bound |
| `.pi/references/tenstorrent/tt-metal-docs/tt_metal/hw/inc/api/dataflow/dataflow_api.h` | what `noc_async_read` / `noc_async_read_barrier` / `cb_push_back` / `get_write_ptr` actually issue | yes |
| `.pi/references/tenstorrent/tt-metal-docs/tt_metal/hw/inc/internal/tt-1xx/blackhole/noc_nonblocking_api.h` | read-issue and flush bookkeeping | yes |
| `.pi/references/tenstorrent/tt-metal-docs/tt_metal/hw/inc/internal/tt-1xx/risc_common.h` | `invalidate_l1_cache()` on BH | yes |
| `.pi/references/tenstorrent/tt-metal-docs/tt_metal/hw/inc/api/compute/bcast.h` | the compute side's unpack path for the scalar CBs | yes |
| `.pi/references/tenstorrent/tt-low-level-documentation/data_movement_doc/general/dm_best_practices.md` | barrier batching discipline | yes |
| pinned `sfpi-gcc` | `instruction-latency` authority | not needed — the change adds no hand-written Tensix/SFPU sequence (see F15) |

## What changed

Parent (`step_reader.cpp:30-39`):

```cpp
template <typename Accessor>
FORCE_INLINE void read_element(const Accessor& src, uint32_t tile, uint32_t row, uint32_t col, uint32_t scratch_l1,
                               uint32_t dst_l1) {
    const uint32_t offset = element_offset(row, col);
    const uint32_t aligned = offset & ~63u;
    noc_async_read(src.get_noc_addr(tile) + aligned, scratch_l1, 64);
    noc_async_read_barrier();
    *reinterpret_cast<volatile uint32_t*>(dst_l1) =
        *reinterpret_cast<volatile uint32_t*>(scratch_l1 + (offset - aligned));
}
```

Candidate (`step_reader.cpp:30-41`):

```cpp
template <typename Accessor>
FORCE_INLINE void stage_element(const Accessor& src, uint32_t tile, uint32_t row, uint32_t col, uint32_t scratch_l1) {
    const uint32_t offset = element_offset(row, col);
    const uint32_t aligned = offset & ~63u;
    noc_async_read(src.get_noc_addr(tile) + aligned, scratch_l1, 64);
}

FORCE_INLINE void store_element(uint32_t scratch_l1, uint32_t row, uint32_t col, uint32_t dst_l1) {
    const uint32_t offset = element_offset(row, col);
    const uint32_t aligned = offset & ~63u;
    *reinterpret_cast<volatile uint32_t*>(dst_l1) =
        *reinterpret_cast<volatile uint32_t*>(scratch_l1 + (offset - aligned));
}
```

and in the instance loop, parent `:150-154`:

```cpp
        read_element(prep, (a_column + head) / 32, batch, (a_column + head) % 32, scratch, get_write_ptr(cb_a));
        read_element(prep, (b_column + head) / 32, batch, (b_column + head) % 32, scratch, get_write_ptr(cb_b));
        read_element(dt, head / 32, 0, head % 32, scratch, get_write_ptr(cb_dt));
        read_element(na, head / 32, 0, head % 32, scratch, get_write_ptr(cb_na));
        noc_async_read_barrier();
```

candidate `:152-163`:

```cpp
        // Stage the four scalar reads behind the same barrier as the row and state reads; each
        // scalar previously paid its own NOC round trip.
        stage_element(prep, (a_column + head) / 32, batch, (a_column + head) % 32, scratch);
        stage_element(prep, (b_column + head) / 32, batch, (b_column + head) % 32, scratch + 64);
        stage_element(dt, head / 32, 0, head % 32, scratch + 128);
        stage_element(na, head / 32, 0, head % 32, scratch + 192);
        noc_async_read_barrier();
        store_element(scratch, batch, (a_column + head) % 32, get_write_ptr(cb_a));
        store_element(scratch + 64, batch, (b_column + head) % 32, get_write_ptr(cb_b));
        store_element(scratch + 128, 0, head % 32, get_write_ptr(cb_dt));
        store_element(scratch + 192, 0, head % 32, get_write_ptr(cb_na));
```

The observable deltas are exactly three: **(a)** each scalar's NOC destination moved from the
scalar's CB page to one of four slots in the scratch page, **(b)** three of the four
`noc_async_read_barrier()` calls disappeared, and **(c)** the four RISC stores now sit
*after* the barrier instead of one-per-barrier *before* it.

## Findings

| # | Audit class | Question | Verdict | Primary evidence |
| --- | --- | --- | --- | --- |
| F1 | noc-sync | Are the NOC read address, size and alignment unchanged? | **pass** | `step_reader.cpp:32-34` vs parent `:33-35` |
| F2 | noc-sync | Is every staged byte guaranteed arrived before the RISC store reads it? | **pass** | `step_reader.cpp:159-163`; `dataflow_api.h:1723-1749`; `risc_common.h:153-157` |
| F3 | noc-sync | Do the four staged reads now race each other (same scratch page)? | **pass** | `step_reader.cpp:155-158`; slots are 64 B apart and disjoint |
| F4 | noc-sync | Does the higher read count per barrier exceed a NOC limit or deadlock? | **pass** (with cost note) | `noc_nonblocking_api.h:193-195, 514-516`; `dm_best_practices.md` "When NOC command buffers are getting full" |
| F5 | noc-sync | Raw `noc_semaphore_*` / multicast / cross-core handshakes introduced? | **not-applicable** | no `noc_semaphore`/`mailbox`/`fence=nop` token in `step_reader.cpp` or `step_writer.cpp` |
| F6 | dataflow-cb-sync | Do the scalar CB credits stay balanced (reserve/push/wait/pop)? | **pass** | reader `:134-138,169-173`; compute `:148-151,176-179` |
| F7 | dataflow-cb-sync | Is the scratch page's ownership and credit accounting still sound? | **pass** (documented invariant) | reader `:114-115`; compute `:38-64` never names CB 10 |
| F8 | dataflow-cb-sync | Do the staged slot offsets stay inside the reserved page? | **pass, fragile** | `decode_step.py:28-29,100-104,308` |
| F9 | dataflow-cb-sync | Is data durable before the credit now that the trailing fence is gone? | **pass** | `MemoryOrdering.md` "Store then store" row + "processing … Subsequent read-requests will observe the stored value"; `dataflow_api.h:208-223` |
| F10 | dataflow-cb-sync / noc-sync | Can a read now race a later instance's overwrite (2 and 4 instances per core)? | **pass** | `step_reader.cpp:125-170`; `decode_step.py:243-246,259-260,307` |
| F11 | dataflow-cb-sync | TRISC/unit ownership of the scratch page and of the staged slots | **pass** | reader `:114-115,155-163`; compute `:38-64`; writer `:32-33` |
| F12 | noc-sync | NOC destination alignment requirement for the new +64/+128/+192 offsets | **pass** | `noc_parameters.h:383` (`L1_ALIGNMENT`, 16 B); `device_print_dispatch.h:21` |
| F13 | mmio-race / cfg-word-overlap / reconfig-stall | Any MMIO/config-register write in the changed kernel? | **not-applicable** | no `cfg[`/`reg_write`/`cfg_rmw`/`regfile[` token in `step_reader.cpp` |
| F14 | semaphore-handshake / mailbox-sync | Tensix semaphore or RISC↔RISC mailbox use in the changed path? | **not-applicable** | none in `step_reader.cpp`; compute's `tile_regs_*` `MATH_PACK` path untouched |
| F15 | srcreg-bank-sync | Source-register / `UnpackToDest` hazard for the scalar CBs? | **pass** (untouched; pre-existing exact-set choice recorded) | `step_compute.cpp:9-11,42-45`; `bcast.h:35-45,79-88`; `decode_step.py:133` |
| F16 | instruction-latency | Hand-written Tensix/SFPU sequence added? | **not-applicable** | baseline barrier batching, `dm_best_practices.md` "CORRECT - Batch operations, then barrier" |

### F1 — NOC read address, size, alignment unchanged: pass

The parent's read is `noc_async_read(src.get_noc_addr(tile) + aligned, scratch_l1, 64)` at
parent `step_reader.cpp:35`; the candidate's is byte-identical except that `scratch_l1` is now
one of four slots, at candidate `:34`. `element_offset()` and the `& ~63u` rounding are
unchanged (`:17-20`). The store's source expression
`*reinterpret_cast<volatile uint32_t*>(scratch_l1 + (offset - aligned))` (`:41`) is the same
expression the parent used at `:38`, so the byte the scalar CB receives is the same byte of
the same 64 B block. Values, word offsets and destination pages are unchanged, as the change
claims.

Out-of-bounds check on that unchanged read: `element_offset` is at most
`3 * 4096 + (15 * 16 + 15) * 4 = 13308`, rounds to `13248`, and `13248 + 64 = 13312` is well
inside the 4096-element fp32 tile (`decode_step.py:28-29`: `TILE = 32`,
`FP32_TILE_BYTES = TILE * TILE * 4`).

### F2 — staged bytes are arrived before the store reads them: pass

`noc_async_read_barrier()` is documented as blocking until *all* reads this core has enqueued
have completed (`dataflow_api.h:1723-1726`: "waits for all the outstanding enqueued
*noc_async_read* calls issued on the current Tensix core to complete. After returning from
this call the *noc_async_read* queue will be empty"), and it ends with
`invalidate_l1_cache()` (`dataflow_api.h:1748`). On Blackhole that is `asm("fence")`
(`tt-1xx/risc_common.h:153-157`), and per `MemoryOrdering.md` ("When a `fence` instruction
enters the Load/Store Unit, the entire L0 data cache will be flushed") the four loads at
`step_reader.cpp:160-163` therefore cannot be served from a stale L0 line. That matters
exactly because the L0 is not coherent: "The cache is _not_ coherent: if other clients write
to L1, doing so will not invalidate the L0 of any baby RISCV" (same page) — the NOC is such a
client. Because the barrier still precedes the loads, the candidate keeps the parent's
guarantee. This is the one property whose loss would corrupt multi-instance widths, and it
is intact.

### F3 — the four staged reads do not race each other: pass

The four destination slots are `scratch + {0, 64, 128, 192}` (`step_reader.cpp:155-158`) and
each write covers exactly 64 bytes (`:34`), so the written ranges `[0,64)`, `[64,128)`,
`[128,192)`, `[192,256)` are pairwise disjoint. Concurrent NOC read responses into one L1
page are not a hazard at disjoint addresses, and no response writes bytes another one owns.

### F4 — read count per barrier: pass, with the queue consequence stated

At the shipped geometry (`head_k_dim = head_v_dim = 128`, `TILE = 32`) the instance issues
`key_tiles = val_tiles = 4`, `state_tiles = 16`, so the reads outstanding at
`step_reader.cpp:159` are `8 (q) + 8 (k) + 8 (v) + 8 (z) + 16 (state) + 4 (scalars) = 52`. The
parent's peak was 49 (the 48 row/state reads plus the first scalar, flushed by that scalar's
own barrier). The barrier is not a queue-depth test — it compares a NOC status counter with
this core's issued count (`noc_nonblocking_api.h:514-516`:
`return (NOC_STATUS_READ_REG(noc, NIU_MST_RD_RESP_RECEIVED) == noc_reads_num_issued[noc]);`,
incremented once per issued read at `:503`) — so a larger batch cannot mis-count. Command
buffer back-pressure is handled inside the API itself, which waits for a free entry before
issuing (`noc_nonblocking_api.h:193-195` `noc_cmd_buf_ready`, polled at `:874`), and the
documented consequence is a stall, not corruption (`dm_best_practices.md`: "When NOC command
buffers are getting full … the code enters a waiting loop until space becomes available. This
creates backpressure in the system."). The direction of this change — fewer barriers, more
reads per barrier — is the documented correct pattern (`dm_best_practices.md`: "WRONG -
Barrier after every operation kills performance" / "CORRECT - Batch operations, then
barrier").

### F5 — raw cross-core handshakes: not-applicable

`step_reader.cpp` and `step_writer.cpp` contain no `noc_semaphore_*`, no mailbox call, and no
cross-core credit. The reader's only cross-core traffic is `noc_async_read` from
`prep`/`state`/`dt`/`na`/`nw` and `get_noc_addr(MEM_ZEROS_BASE)` (`:26-27,34,49,120,151`), all
reads, so there is no data-before-credit ordering to audit at this layer.

### F6 — scalar CB credit balance: pass

Reader side, per instance: `cb_reserve_back(cb_a, 1) … cb_reserve_back(cb_na, 1)`
(`step_reader.cpp:134-137`) is matched by `cb_push_back(cb_a, 1) … cb_push_back(cb_na, 1)`
(`:169-172`), and `cb_push_back(cb_state, state_tiles)` closes the `state_tiles` reserve at
`:138`. Compute side (`step_compute.cpp:148-151` waits, `:176-179` pops) matches one page
per CB per instance. The change does not move any reserve, push, wait or pop; it only moves
*when* the destination page is filled inside the already-reserved window.

### F7 — scratch-page ownership and credit accounting: pass, invariant recorded

`cb_scratch = 10` (`step_reader.cpp:93`) is reserved once, before the instance loop
(`:114-115`):

```cpp
    cb_reserve_back(cb_scratch, 1);
    const uint32_t scratch = get_write_ptr(cb_scratch);
```

and it is never pushed or popped, in the parent or the candidate. That is sound only because
CB 10 is a **reader-private static workspace**, not a FIFO: `step_compute.cpp:38-64` declares
CBs `0-9` and `11-27` and never names 10, `step_writer.cpp:32-33` owns only `16` and `22`, and
`decode_step.py:308` sizes it `10: 1` page. The change keeps this invariant (it only widens
the used part of the page) but it is now load-bearing for four slots instead of one, so it is
recorded here: **no other kernel may take CB 10, and the reader must keep reserving it once
outside the loop.**

### F8 — slot bounds inside the reserved page: pass, fragile

CB 10's page is 4096 bytes (`decode_step.py:29` `FP32_TILE_BYTES = TILE * TILE * 4` with
`TILE = 32`; `:100-104` sets `total_size = pages * page_bytes` and
`page_size=page_bytes`; `:308` gives it one page). The staged slots end at byte 256
(`step_reader.cpp:158,163`), i.e. 16× inside the page. Nothing in the kernel or the host
asserts that the page is at least 256 B, so a future grid/tile change that shrank an fp32
page below 256 B would silently spill into the next CB. Not a defect of this change; noted
because the change is what makes the page carry four slots.

### F9 — data durable before the credit, now that the trailing fence is gone: pass

This is the one ordering question the diff genuinely changes. In the parent, each scalar's
store was followed by `noc_async_read_barrier()` (`:154`), whose `fence` was the last thing
between the RISC store and `cb_push_back` (`:156-162`). In the candidate the barrier is at
`:159`, the stores are at `:160-163`, and `cb_push_back(cb_a, 1)` follows at `:164` with no
fence between them.

It is still safe, on `MemoryOrdering.md` rather than on the fence:

- Both the tile store (`:160-163`) and the credit store are RISC stores into **L1**
  (`dataflow_api.h:211-212`: `volatile tt_reg_ptr uint32_t* pages_received_ptr = …;
  pages_received_ptr[0] += num_pages;`). A fence is not what makes the pair safe.
- `MemoryOrdering.md`, "Store then store": "If the two stores are to the same region of
  memory, the two requests will be processed in order."
- `MemoryOrdering.md`, "processing": "For write-requests, processing means _actually_ writing
  the value to long-term storage. Subsequent read-requests will observe the stored value."
- Therefore the consumer's `cb_wait_front` can only see the incremented credit after the tile
  store has been processed, at which point any later read of the tile observes it.

The cross-client step (the consumer is T0/T1, not the reader's RISC) rests on that last
sentence not being scoped to the issuing client — tagged `[INFERENCE]`. The removed fence
never provided more than this: per the same page, draining the store queue "merely ensures
that the write requests have been sent on their way … it doesn't ensure that the write
requests have actually reached their final destination and been processed." So the parent's
trailing fence was not the load-bearing guard, and losing it costs nothing.

### F10 — no read can race a later instance's overwrite: pass

Scope: one reader kernel per core (`decode_step.py:266-283` one `KernelDescriptor` over
`core_set`), one instance loop per core
(`step_reader.cpp:125`: `for (uint32_t instance = instance_start; instance < instance_start + instance_count; instance++)`),
and the four slots are re-used by every instance on that core. Three orderings cover it,
and each holds independently of how far compute has progressed:

1. **Within an instance**: the barrier at `:159` precedes the four loads at `:160-163`
   (F2), so the instance's own staged bytes are present when read.
2. **Between the load and the slot's reuse**: the store at `:160-163` obtains its value by a
   RISC load whose read-response must arrive before the store retires (`MemoryOrdering.md`
   load row: "Reached front of retire-order queue, read-response obtained from memory
   subsystem"), and the next instance's `stage_element` NOC command is issued later still by
   the same thread. So the slot's next overwrite cannot precede the value capture.
3. **Across instances into the destination CBs**: `cb_a … cb_na` hold one page each
   (`decode_step.py:307`: `4: 1, 5: 1, 6: 1, 7: 1`), so the next instance's
   `cb_reserve_back` cannot return until compute has popped the page the previous instance
   pushed. The reader can never be more than the current instance ahead, and the scratch
   slots need no such gate because their reuse is already ordered by (2).

(`decode_step.py:259-260` gives each core a contiguous run
`start = index * per_core`, `count = min(per_core, instances - start)`, so instances on one
core differ in `head`, never in which slot set they own.)

### F11 — TRISC/unit ownership: pass

The scratch page is touched only by the reader: written by NOC read responses from the
reader's own `noc_async_read` (`step_reader.cpp:155-158`) and read by the reader's own RISC
loads (`:160-163`). Neither compute (`step_compute.cpp:38-64`) nor the writer
(`step_writer.cpp:32-33`) names CB 10, so no Tensix thread and no other RISC reaches it. The
scalars reach compute only through `cb_a … cb_na`, and only inside the credit window
(F6). The change moves no ownership boundary.

### F12 — destination alignment of the new offsets: pass

A NOC transfer into L1 needs `L1_ALIGNMENT` alignment, which is 16 B
(`tt-1xx/blackhole/noc_parameters.h:383`; the comment in
`tt_metal/impl/dispatch/kernels/device_print_dispatch.h:21` reads
`NOC_L1_TO_L1_ALIGNMENT = L1_ALIGNMENT; // 16`). `get_write_ptr` returns the CB's byte
`fifo_wr_ptr` (`dataflow_api.h:323-327`), which is at least that aligned, and adding 64,
128 or 192 bytes preserves 16 B alignment. No new alignment requirement is introduced.

### F13 — MMIO / config-register / reconfig: not-applicable

The changed kernel performs no MMIO and no config-register write: `step_reader.cpp` contains
no `cfg[`, `reg_write`, `cfg_rmw` or `regfile[` token. Its only stores are the four
`volatile uint32_t*` L1 stores at `:40-41` into CB pages, and its only other writes are
`cb_*` credit/L1-pointer updates. So the MMIO-race, cfg-word-overlap and reconfig-stall
classes have no site in the diff. Compute's config writes (via `reconfig_data_format`,
`unary_bcast_init`, …) are untouched by the commit — `step_compute.cpp` is byte-identical.

### F14 — Tensix semaphores / mailboxes: not-applicable

No semaphore or mailbox primitive appears in the changed kernel (F5). Compute's `tile_regs_*`
`MATH_PACK` handshake (`step_compute.cpp:154-173`) is in the untouched file, and the change
does not alter any operand the compute thread builds from it.

### F15 — source-register / `UnpackToDest` hazard: pass, untouched

The change leaves the scalar CBs' format, dtype, tile index and written bytes identical, so the
unpacker is handed the same bits from the same page; compute's SrcA/SrcB bank bookkeeping is
untouched (the file is byte-identical). The compute kernel's own header records which pages
bypass the source registers (`step_compute.cpp:9-11`: "The recurrent state never passes
through the 19-bit source registers … Only matmul operands … are rounded to tf32. CBs marked
'exact' below are unpacked to DEST in fp32"), and the scalar CBs `4-7`
(`step_compute.cpp:42-45`) are **not** in that marked set (`decode_step.py:133`
`EXACT_STEP_CBS = (2, 3, 8, 9, 11, 12, 20, 26, 27)`). The broadcast they are consumed by
(`step_compute.cpp:116-121`) does take the API's 32-bit unpack-to-dest branch
(`bcast.h:35-45` and the same decision inside `unary_bcast` at `:79-88`: "32bit formats are
implemented using unpack to dest, since SrcB is only 19bits wide"), but whether the
per-CB `unpack_to_dest_mode` marks those four pages Fp32 is the host's pre-existing choice and
this commit does not change it. So: **not a hazard introduced or touched here**; if the four
scalars should be marked exact it is a separate numeric decision, and V11
(`evidence/step-kernel-scalar-staging.md`) is the record that guards the values actually
consumed. No bank-flip, dvalid or single-owner question arises, because neither side of the
unpack→math handshake moves.

### F16 — instruction latency: not-applicable

The added code is C++ loads/stores plus one extra `noc_async_read` issue — no hand-written
`TTI_*`/`TTI_SFP*` sequence, so the `sfpi-gcc` latency authority has nothing to arbitrate. The
barrier *count* change is the documented batching form (`dm_best_practices.md`), not a
pipeline padding question.

## Two and four instances per core (widths 8 and 16)

`decode_step.py:243-246` opens `instances = batch * g.num_v_heads` and spreads them
`per_core = -(-instances // (grid.x * grid.y))` over `cores`, so at `num_v_heads = 24` and the
Blackhole compute grid of roughly 110 cores: width 1 → 24 instances, one per core; width 8 →
192 instances, two per core; width 16 → 384 instances, four per core. Widths 8 and 16 are
therefore the only widths that re-enter the instance loop on one core, and they are the only
widths that exercise slot reuse (F10) and repeated use of the one-page scalar CBs.

Auditing that case specifically:

- The staged slots are re-used, but (F10.2) makes their reuse ordered by the reader's own
  load-before-reissue, so there is **no** read/overwrite race that only appears at two
  instances per core.
- The only hazard that multi-instance widths would have exposed is the L0 data cache serving
  a **previous** instance's staging to the current load — the same slot address is loaded
  every instance, and the L0 is not coherent with NOC writes (`MemoryOrdering.md`). The
  barrier at `step_reader.cpp:159` flushes it before each instance's loads (F2), so this is
  discharged rather than assumed. Had the change staged the reads *without* a following
  `fence`/`invalidate_l1_cache`, this would have been a silent per-instance corruption at
  widths 8 and 16 only.
- The four-slot layout changes nothing else at those widths: instances on one core differ in
  `head`, hence in `(a_column + head) / 32` and `% 32` (`:155-156`) and in `head / 32`,
  `head % 32` (`:157-158`), so each instance stages from and stores to different elements of
  its own pages.

## What this establishes, and what it does not

- Establishes, by source reading: commit `ed2f4a1` changes one kernel (`step_reader.cpp`) and
  no other, preserves the NOC read addresses/sizes/values and both destination pages, keeps a
  read-completion barrier plus an L0-flushing `fence` ahead of every staged load, keeps every
  CB credit pair balanced, keeps CB 10 reader-private and inside its 4096 B page, and
  introduces no MMIO, config, semaphore, mailbox, source-register or hand-written
  instruction sequence. All applicable audit classes pass; four are not applicable.
- Records the invariant the change makes load-bearing: CB 10 is a reader-only static
  workspace reserved once outside the instance loop and never pushed, and the reader may not
  grow past 256 B of its page without asserting that the page is that large (F7, F8).
- Does **not** claim silicon evidence. There is none in this file: no device execution, no
  simulator run, no trace, no timing, no occupancy, and no admission movement. The parity and
  timing record for this commit is `evidence/step-kernel-scalar-staging.md` (V11), which this
  audit does not re-derive or extend.
- **Coverage bound:** the pinned ISA snapshot has no `BlackholeA0/NoC/Ordering.md` (only
  `WormholeB0/NoC/Ordering.md` exists), so BH NoC ordering is grounded here on the BH RISC
  `MemoryOrdering.md`, on what the dataflow API issues (`dataflow_api.h`,
  `noc_nonblocking_api.h`) and on `dm_best_practices.md`. That bound touches F2 (the `fence`
  and L0 half is grounded on `MemoryOrdering.md`, which is present) and F9, whose cross-client
  step is tagged `[INFERENCE]`. Nothing else in the table depends on BH NoC packet ordering,
  because the change issues only unicast reads and no credit.
