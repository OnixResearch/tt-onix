# V6: bounded Blackhole kernel hazard audits over both changed factories and their kernels

Recorded 2026-09-11 for change `lift-gdn-grid-instance-cap`, task V6.

## Scope

Audited from the shipped variant runtime
`/nix/store/6ccynlvv…-tt-metal-0.77.0/libexec/tt-metalium/ttnn/cpp/ttnn/operations/transformer/`:

- `gated_delta_decode/device/kernels/`: `compute/gated_delta_decode.cpp` (212
  lines), `dataflow/reader_gated_delta_decode.cpp` (108),
  `dataflow/writer_gated_delta_decode.cpp` (67);
- `chunk_gated_delta_rule/device/kernels/`: `dataflow/reader_chunk_gdn_scan.cpp`
  (116), `dataflow/writer_chunk_gdn_scan.cpp` (82), `compute/chunk_gdn_scan.cpp`
  (185);
- both program factories via `pkgs/tt-metal/gdn-decode-grid-chunk.patch` and
  `pkgs/tt-metal/gdn-prefill-scan-chunk.patch`.

The audits: CB credit balance (dataflow-cb-sync), NoC data-before-credit
ordering (noc-sync), inter-thread synchronization surfaces (semaphore,
mailbox, MATH_PACK/UNPACK handshakes), config-write hazards (mmio-race,
reconfig-stall, cfg-word-overlap), hand-written instruction latency, and the
SrcA/SrcB/Dst shared-register surface.

## Findings

**No blocking hazard found.** Per-kernel detail:

**CB credit balance holds per instance and across instances.** Every
producer window in both readers is `reserve -> async reads -> barrier ->
push`; every consumer window in both writers is `wait -> writes -> barrier ->
pop`; compute waits and pops exactly the counts it consumes (q last, after
the output matmul; k after transpose; v after delta; beta after
outer_scaled; g after g_exp; state after state_decay). The decode
state/output cross-thread handoff keeps its pre-change ordering: compute
reads `state_new` before pushing output, and the writer waits output before
state, so the writer's pop cannot race compute's read.

**The scan state ping-pong returns to zero at instance end.** `cb_s2`/`cb_s3`
alternate by chunk parity; the final chunk of every instance writes
`cb_final`, and chunk `c+1` consumes what chunk `c` produced, so no credit
survives an instance boundary and the next instance's `cur_S = cb_S` starts
clean. The reader-pushed `cb_S` is pushed once per instance and consumed at
chunk 0 of the same instance.

**NoC ordering is correct.** Both readers complete `async_read_barrier`
before `push_back`; both writers complete `async_write_barrier` before
`pop_front`. No remote CB credits, no noc semaphores, no multicast.

**No semaphores, mailboxes, or raw register writes anywhere in the six
kernels.** All inter-thread synchronization is CB credits; all config goes
through the packaged API (`pack_reconfig_data_format`, `reconfig_data_format`,
`matmul_init`, `add_init`, `init_sfpu`, ...), so no mmio-race, cfg-word, or
mailbox surface exists to audit. The SFPU exp sequence and the matmul init
sequences are unchanged from the pre-change kernel and are re-initialized
inside the instance loop, so each instance re-establishes its config before
use (reconfig-stall satisfied by the API at stage boundaries between full CB
windows).

**`compute_kernel_hw_startup` stays outside both instance loops**, matching
its per-core, once-per-kernel semantics.

**Artifact (harmless, recorded): the decode reader carries a redundant
pre-loop `reserve_back` block.** `cb_reserve_back` in this runtime is a
stateless free-space wait (`dataflow_api.h:404`: it recomputes
`fifo_num_pages - (pages_received - pages_acked)` and does not track prior
reservations), so the pre-loop reserves pass trivially, consume nothing, and
write nothing. They are dead weight left over from the pre-loop structure of
the one-instance kernel, not a deadlock and not an overwrite: each instance's
writes occur after the previous instance's `push_back` advanced the write
pointer. This is consistent with the B=1 silicon pass under `full`. Removing
the block is cosmetic and deliberately deferred to keep the patch stable
through V8/V9.

## Non-claims

Static audit of the shipped kernel sources. It cannot prove the absence of a
hardware-timing hazard that only manifests at two instances per core; V8 and
V9 execute exactly that case on silicon.
