# Fused WKV kernel — bounded hazard and performance audits

## Scope

This audit covers the final Phase 2 source in these files:

- `pkgs/rwkv7-p150x2/kernel/fused_wkv_reader.cpp`;
- `pkgs/rwkv7-p150x2/kernel/fused_wkv_compute.cpp`;
- `pkgs/rwkv7-p150x2/kernel/fused_wkv_writer.cpp`;
- `pkgs/rwkv7-p150x2/src/fused_wkv_program.cpp`; and
- `pkgs/rwkv7-p150x2/src/main.cpp`.

The fixed profile has 12 heads, a head width of 64, and a maximum batch of 16.

`SAFE` means that the reviewed source has no remaining candidate in the stated scope.
It does not mean that physical hardware accepted the kernel.

## Authorities

The audit used these package-matched or pinned sources:

- TT-Metal `v0.74.0-rc6` compute, dataflow, CB, and NoC APIs;
- TT-ISA commit `b7738d9ac14a34a4033d60dde9463466b23082e1`;
- TT low-level documentation commit `cf344d2dda4cb9d06451d0cc62e9a12c7f1b9240`;
- SFPI `7.64.0`; and
- `sfpi-gcc` commit `b64d7a407259135b833e370fcee46686bee21918`.

The low-level documentation defines the required NoC barrier ordering.
The package CB API defines reserve, publish, wait, and release cycles.

## Corrected audit findings

The bounded audit found two correctness defects.
Both defects are fixed in the reviewed source.

### Reduction reconfiguration order

The first source initialized a binary operation before it ended reduce mode.
The package API requires `reduce_uninit(input_cb)` first.

The source now calls `reduce_uninit(input_cb)` before `add_binary_tile_init()`.
A negative source-contract test rejects the old order.

### Correction-scalar CB cycle

CB 29 publishes a two-tile row and then a one-tile scalar.
A two-tile capacity cannot complete that repeated pointer cycle.

The CB now has a three-tile capacity.
A negative source-contract test rejects the old two-tile capacity.

## Nine hazard audit verdicts

### 1. MMIO race — SAFE

The kernels contain no direct config-array writes, GPR writes, `WRCFG`, raw `TTI_*`, or inline assembly.
Reader padding writes target reserved L1 CB storage after a completed NoC read.

### 2. Reconfiguration stall — SAFE after correction

All format changes use the package compute APIs.
The Blackhole LLK paths drain the required unpack, math, or pack unit before config changes.

The reduce-mode ordering defect is fixed as described above.
No direct config rewrite bypasses those APIs.

### 3. Config-word overlap — SAFE

The kernels do not write backend config words directly.
They do not use full-word writes that can clobber sibling fields.

### 4. Semaphore handshake — SAFE

Each active core owns one local completion semaphore.
Only its writer kernel updates that semaphore.

The writer publishes the final owned-work count after all output writes complete.
No kernel waits on that publication, so it cannot form a wait cycle.

### 5. Mailbox synchronization — SAFE

The fused kernels do not use RISC mailboxes.
There is no push, pop, address-direction, or FIFO-balance candidate.

### 6. Dataflow CB synchronization — SAFE after correction

Every raw input uses reserve, completed NoC read, push, wait, and pop ordering.
Every output uses reserve, pack, push, wait, completed NoC write, and pop ordering.

All single-shape CBs complete exact pointer cycles.
CB 29 now completes its `2 + 1` tile cycle with capacity 3.

The maximum-batch simulator covered the full work distribution.
It included repeated work on multiple cores and completed 192 work items per step.

### 7. Source-register bank synchronization — SAFE

The compute kernel uses tile-register and compute APIs.
It does not issue raw DVALID, bank-flip, SrcA, SrcB, Dst, or LReg controls.

Each tile-register region has explicit acquire, commit, wait, and release calls.
Package LLK handshakes own the unpack, math, and pack bank transitions.

### 8. NoC synchronization — SAFE

Reader data is complete before padding sanitation and CB publication.
Writer data is complete before CB release and completion publication.

Each work item owns disjoint output and next-state pages.
The kernels do not use remote NoC credits, multicast fan-out, or cross-core NoC semaphores.

The final writer barrier was redundant because each `write_tiles` call already blocks.
The redundant barrier is removed.

### 9. Instruction latency — SAFE for the bounded source

The fused sources contain no hand-written Tensix or SFPU sequence.
They use sfpi and package compute APIs, so the matching compiler owns scheduling.

The matching Blackhole TRISC1 cache artifact is:

`tt-metal-cache/227660415213024366/kernels/fused_wkv_compute/17936049079751221546/trisc1/trisc1.elf`

The audit used a source-matched TRISC1 disassembly of that ELF.

The rsqrt path expands to the Blackhole compatibility sequence.
The ELF contains no `SFPARECIP`, `SFPLOADMACRO`, `SFPNOP`, or raw NOP mnemonic.

No emitted sequence gives evidence for a missing manual delay.
This verdict does not replace a physical numerical test.

## Cross-class JOIN audit — SAFE

The JOIN pass checked dependencies between the separate audits.

- Reader NoC completion precedes padding sanitation and CB credit.
- Compute CB credit precedes writer access.
- Writer NoC completion precedes CB release and completion publication.
- Package reconfiguration drains the unit that owns each changed format.
- The corrected reduce order ends reduce mode before binary initialization.
- Unique work ownership gives one writer for every state and output page.
- Exact CB cycles remain valid when one core handles repeated work items.

No verdict depends on an unproved invariant from another hazard class.

## Performance audit

The audit removed one source-proven redundant writer barrier.
No speed claim follows from that source change.

Three measurement candidates remain:

1. The reader completes one state read and ten vector reads separately.
2. The writer completes output and next-state writes separately.
3. The correction path stores and reloads several two-tile scratch vectors.

These candidates can reduce overlap or increase L1 traffic.
Only matching physical profiling can rank them or prove a gain.

The compiled kernel has no source-authored NOP sequence to remove.
The deprecated transpose and matmul wrappers remain package-supported.
Their replacement needs a separate parity and performance change.

## Simulator evidence

Retained receipt:

`cairn/changes/fuse-rwkv7-wkv-kernel/evidence/phase2-fused-wkv-simulator-receipt.json`

BLAKE3:

`16c3edd79d63fd83e34fbe9f64c6e20662b32e0e29827f6eab2ced20d8fc6e88`

The Blackhole simulator ran batch 16, 12 heads, and two recurrent steps.
All four maximum absolute errors are `0.0`.

The receipt has outcome `fused_wkv_corrected_fixture_pass`.
It covers normalization, `r_k` correction, gate multiplication, and BF16 output.

The receipt also records these non-claims:

- `simulated: true`;
- `physical_hardware: false`;
- `performance_claim: false`; and
- `production_physically_admitted: false`.

## Admission result

The bounded source hazard verdict is `SAFE` after the two corrections.
The simulator fixture passes at the maximum supported batch.

Production admission was closed when this hazard audit completed.
Later physical parity and performance evidence opened dispatch only under the measured CPU and FIFO scheduler contract.
