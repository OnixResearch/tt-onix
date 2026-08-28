# Fused WKV Blackhole hazard audit

## Scope

This audit covers all changed `pkgs/rwkv7-p150x2/kernel/fused_wkv*.cpp` files and their host descriptors in `pkgs/rwkv7-p150x2/src/fused_wkv_program.cpp`.

The target architecture is Blackhole A0. Wormhole and Quasar are outside the launch contract and were not used to infer Blackhole behavior.

## Grounding

- Package-matched TT-Metal revision: `v0.74.0-rc6`, commit `c49bb7625e6a495337483353a5228339fece2c11`.
- Blackhole ISA checkout: commit `b7738d9ac14a34a4033d60dde9463466b23082e1`.
- Blackhole memory ordering: `BlackholeA0/TensixTile/BabyRISCV/MemoryOrdering.md`.
- Package dataflow API: `tt_metal/hw/inc/api/dataflow/dataflow_api.h`.
- Blackhole NoC implementation: `tt_metal/hw/inc/internal/tt-1xx/blackhole/noc_nonblocking_api.h`.
- Pinned SFPI version: `7.64.0`.
- Pinned `sfpi-gcc` revision: `b64d7a407259135b833e370fcee46686bee21918`.
- DeepWiki access to `tenstorrent/tt-isa-documentation` was available during source preflight.
- Internal Confluence was unavailable. The Blackhole `NoC/Ordering.md` page is also absent from the pinned ISA checkout. Therefore, this report does not infer unsupported NoC semantics from document silence.

## Raw reports

### 1. MMIO race

**Verdict: SAFE — no candidates.**

The changed kernels do not contain raw `cfg[...]`, `reg_write`, `cfg_rmw`, `regfile[...]`, or hand-issued MOP/replay writes. Compute synchronization uses package compute APIs. Dataflow synchronization uses CB and NoC APIs.

Candidate count: `0`.

### 2. Reconfiguration stall

**Verdict: SAFE — no direct config-write candidates.**

The changed kernels do not directly rewrite ALU, THCON, address-modifier, or stride configuration words. Compute API initialization and reconfiguration calls stay inside one compute thread. Every compute stage waits for tile-register completion before packing and releasing registers.

Candidate count: `0` direct config writes.

### 3. Config-word overlap

**Verdict: SAFE — no shared full-word writers.**

The changed sources do not directly write shared backend configuration words. There is no cross-thread full-word write or RMW on an ALU-format, accumulator-control, rounding, ReLU, THCON, or address-modifier word.

Candidate count: `0`.

### 4. Semaphore handshake

**Verdict: SAFE for in-scope inter-thread ownership.**

The kernels do not use Tensix `SEMINIT`, `SEMWAIT`, `SEMPOST`, `SEMGET`, `t6_semaphore_*`, or `t6_mutex_*` handshakes. The completion word is a writer-owned local L1 semaphore. Each host descriptor creates it with `kCompletionSemaphoreId`, and each writer sets it after all assigned work completes.

Candidate count: `7` host completion descriptors and `7` writer signals. All are balanced by construction.

### 5. Mailbox synchronization

**Verdict: SAFE — no candidates.**

The changed kernels do not use `mailbox_write`, `mailbox_read`, `mailbox_not_empty`, or `TENSIX_MAILBOX*`.

Candidate count: `0`.

### 6. Dataflow CB synchronization

**Verdict: SAFE.**

Every reader follows this order:

1. reserve output CB pages;
2. issue bounded page reads;
3. wait with `noc_async_read_barrier()`;
4. sanitize or materialize local data when required; and
5. publish pages with `cb_push_back()`.

Every compute stage follows this order:

1. wait for all input pages;
2. reserve output pages;
3. acquire tile registers;
4. compute, commit, and wait;
5. pack and release tile registers;
6. publish the output pages; and
7. pop each input exactly once after its last use.

Every dedicated diagnostic output uses `c16`. Compute publishes it. Only the writer waits for and pops it. The full split recurrence uses `c22` through `c26` as compute-owned intermediate CBs. The full writer consumes only the final `c9` state in split mode. It does not tap `c8`, `c23`, or another compute-internal CB.

Each CB has one credit consumer. No read-pointer or acknowledgement ownership transfers between compute and writer.

Candidate count: all reserve/push/wait/pop sites in the `fused_wkv*.cpp` manifest. Finding count: `0`.

### 7. Source-register bank synchronization

**Verdict: SAFE — no raw bank-handshake candidates.**

The changed kernels do not issue raw `SETDVALID`, `CLEARDVALID`, `MOVD2A`, `MOVA2D`, or `MOVB2D`. They use package compute APIs and the tile-register acquire/commit/wait/release protocol. No Dst or LReg value is shared directly across RISC threads.

Candidate count: `0` raw sites.

### 8. NoC synchronization

**Verdict: SAFE for the explicit barrier chains; no cross-core semaphore candidates.**

Readers wait for all issued reads before publishing CB credits. Writers wait for all issued non-posted writes before popping output pages and setting completion. The package dataflow API implements `noc_async_read_barrier()` by polling the read-flushed condition and `noc_async_write_barrier()` by polling the non-posted-write-flushed condition.

The writers call local `noc_semaphore_set`, not a remote set or increment. No kernel uses `noc_semaphore_wait`, `noc_semaphore_inc`, multicast signaling, or remote CB credits.

Because Blackhole NoC ordering documentation is incomplete and internal Confluence was unavailable, this report does not claim a broader undocumented write-to-signal rule. The code does not need that rule: it uses an explicit write barrier before the local completion signal.

Candidate count: all reader barriers, writer barriers, and `7` completion signals. Finding count: `0`.

### 9. Instruction latency

**Verdict: SAFE — no hand-written instruction candidates.**

The changed kernels do not contain raw `TTI_*` or `TT_OP_*` instruction streams. The SFPU multiply and add paths use package compute APIs and compile through the pinned SFPI toolchain. The audit found no hand-scheduled dependent instruction sequence that needs a manual NOP decision.

Candidate count: `0` raw instruction sequences.

## Cross-class JOIN

| Safety dependency | Joined evidence | Result |
|---|---|---|
| Reader data must be complete before CB credit | Each read sequence ends with `noc_async_read_barrier()` before `cb_push_back()` | Discharged |
| Compute output credit must have one consumer | Dedicated outputs use `c16`; only the writer pops. Split intermediates remain compute-owned | Discharged |
| Writer completion must mean device writes completed | Each write sequence calls `noc_async_write_barrier()` before CB pop, completion count, and `noc_semaphore_set()` | Discharged |
| CB pop must not free data while NoC still reads it | Writer pop follows the write barrier | Discharged |
| Tile-register data must be ready before packing | Every compute stage commits and waits before pack; release follows pack | Discharged |
| Split fidelity boundary must not share compute state | HiFi4 recurrence and HiFi2 readout are separate programs with synchronized tensor readback | Discharged |
| Source-bank safety must not rely on a missing custom semaphore | Package compute APIs own source-bank handoff; no raw DVALID or bank-flip operations exist | Discharged |
| Instruction-latency safety must use the pinned compiler path | No raw instruction stream exists; device compilation uses the package SFPI compiler | Discharged |

## Emergent findings

No emergent race was found.

No per-class finding was dropped or downgraded. There were no non-safe per-class findings to reconcile.

## Coverage bound

This is an exhaustive source audit for the declared fused WKV kernel and host-descriptor scope. It is not an audit of all TT-Metal kernels or all architectures.

Blackhole NoC documentation is incomplete, and internal Confluence was unavailable. No verdict relies on a missing document. Unsupported general NoC semantics remain unclaimed.
