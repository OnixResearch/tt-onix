# Bounded Blackhole kernel hazard audit

## Scope and authority

This audit covers only the new `gated_delta_decode` operation and its three device kernels. It is not a full TT-Metal audit.

The user approved this bounded Blackhole-only scope on 2026-08-27. The sources were:

- DeepWiki for `tenstorrent/tt-isa-documentation`.
- Local ISA checkout `b7738d9ac14a34a4033d60dde9463466b23082e1`.
- SFPI 7.70 source `6659aeb9e350dd8eef69edbdfa4f963759b04c44` and GCC submodule `6ffe1f327045f30dadd241259463f4ce83eb0bd9`.
- Package-matched TT-Metal 0.74 dataflow, compute API, and Blackhole LLK source.

The dedicated TT ISA and Confluence MCP servers were unavailable. Confluence is not required for this Blackhole-only scope. The pinned ISA tree has no `BlackholeA0/NoC/Ordering.md`. This audit abstains from unresolved NoC ordering claims.

## Source compatibility finding

**Resolved build blocker.** The first source used `matmul_reconfig_and_init`, which TT-Metal 0.74 does not provide. Host compilation did not detect this JIT-only symbol.

The kernel uses package-matched `matmul_init` at all three matrix operations. It also uses `transpose_init` and `transpose_tile` instead of deprecated aliases.

The first physical JIT attempt exposed a second source-only defect. The compute kernel had only the umbrella header, so TRISC-specific APIs were not declared. The repair uses the same explicit API headers as the package-matched chunk GDN kernel.

That review also found a missing mandatory `compute_kernel_hw_startup` call. The kernel now calls it exactly once, before every other compute API call.

The second physical JIT attempt found three remaining newer init names. The package-matched replacements are `mul_tiles_bcast_scalar_init_short`, `sub_tiles_init`, and `add_tiles_init`. The startup call now uses `SrcOrder::Reverse`, as required by `matmul_init`.

An exact compiler replay passed for all three TRISCs against the corrected source-tree file. A replay against the rebuilt installed source then found that the new-file patch hunk still declared 207 lines. Nix therefore omitted the 208th line, which was the closing brace.

The hunk count is now corrected. Package `/nix/store/acw86r5sd0yi20p55saa1sy1kik7s2h4-qwen36-0.1.0` passed the device-free checks and a new installed-source line, terminator, and API-name guard. Exact compilation of its installed source passed for unpack, math, and pack TRISCs.

## Raw audit reports

### 1. MMIO race

Candidate count: 0.

The changed kernels contain no raw CONFIG, GPR, MOP, or register-file writes. `compute_kernel_hw_startup` performs API-owned initial configuration before execution starts. Verdict: **no custom MMIO race found**.

### 2. Reconfiguration stall

Direct candidate count: 1 initial API configuration.

The kernel calls `compute_kernel_hw_startup<SrcOrder::Reverse>(cb_k, cb_state_decay, cb_v_read)` exactly once. It is the first compute API call and establishes the source order required by matrix operations. The full `init_sfpu` call reconfigures the first unary operation. All involved CBs use the same FP32 tile geometry, so the documented asymmetric-tile gap does not apply.

Operation changes use package-matched `matmul_init`, `mul_tiles_bcast_scalar_init_short`, `sub_tiles_init`, `transpose_init`, and `add_tiles_init` calls.

Verdict: **safe after repair** for the package API paths in scope. Both failed physical attempts stopped during JIT and did not execute the kernel.

### 3. Configuration-word overlap

Candidate count: 0.

The changed source contains no direct CONFIG writes or masked/full-word RMW operations. Verdict: **not applicable in the declared source scope**.

### 4. Semaphore handshake

Explicit semaphore or mutex candidate count: 0.

The compute kernel uses the standard destination-register handshake. Every operation has a complete `tile_regs_acquire` → `tile_regs_commit` → `tile_regs_wait` → `tile_regs_release` sequence. There are no branches or early returns between these calls.

For the fixed 128 by 128 head geometry, each head executes 81 balanced destination windows. Verdict: **safe for the declared control flow**.

### 5. Mailbox synchronization

Candidate count: 0.

The changed kernels contain no mailbox reads, writes, or readiness checks. Verdict: **not applicable**.

### 6. Dataflow circular-buffer synchronization

Candidate count: 15 circular buffers.

- Reader inputs `c_0` through `c_5` reserve their full capacity, issue reads, run one `async_read_barrier`, and then push the same count.
- Compute intermediates reserve and push the same count before each consumer waits.
- Each buffer has one pointer-updating producer and one pointer-updating consumer.
- Every reserve and wait count equals its factory capacity.
- Writer outputs run `async_write_barrier` before their matching pop.

`c_13` has one pointer-updating consumer, the writer. Compute reads it without popping. The writer first waits for `c_16`, which compute pushes only after `q @ state_new` completes. This orders the state write and pop after compute's final state read.

Verdict: **safe** for balance, capacity, pointer ownership, and explicit transfer completion.

### 7. Source-register bank synchronization

Direct candidate count: 0.

The source has no raw `SETDVALID`, `CLEARDVALID`, bank-pointer operation, or direct Dst/LReg move. It uses standard compute APIs inside balanced destination windows. Verdict: **no direct custom bank hazard found**.

### 8. NoC synchronization

Raw semaphore or multicast candidate count: 0.

The reader uses one read barrier before publishing input CB credits. The writer uses one write barrier for output and one for state before releasing the matching CB credits. No separate NoC semaphore or atomic signal exists.

Verdict: **safe for the explicit transfers**. No broader Blackhole packet-ordering claim is made.

### 9. Instruction latency

Raw instruction candidate count: 0.

The kernel contains no `TTI_*`, inline assembly, or direct opcode stream. The exp path and matrix operations use package compute APIs. SFPI 7.70 GCC scheduling source was reachable, but no custom raw sequence required a latency verdict.

Verdict: **not applicable to custom raw instructions**.

## Cross-class join

The only cross-class seam is `c_13` state ownership. CB safety depends on the compute-to-writer order. That order is discharged by `c_16`: compute pushes output after its final state read, and the writer waits for output before it writes or pops state.

No emergent race was found. No per-class finding was dropped or downgraded.

## Performance review

The kernel intentionally trades local work for one host operation launch.

One exact-order optimization remains available: process up to `compute_kernel_lib::DEST_AUTO_LIMIT` independent tiles per destination window. The current one-tile windows execute 81 destination handshakes per head. This change can reduce handshake count without reordering arithmetic inside each tile.

Moving beta multiplication before the outer-product matrix operation would remove state-sized work. It changes fp32 rounding order, so it is only a suggestion that needs long-decode parity evidence.

## Audit verdict

The repaired source passed a fresh package build and exact installed-source TRISC compilation. It still needs physical execution and numerical comparison. This audit does not replace device execution, trace replay, or production admission.
