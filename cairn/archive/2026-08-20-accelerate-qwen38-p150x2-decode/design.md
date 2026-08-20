## Context

The current P150x2 service uses eager decode and host sampling. Its measured batch-one baseline is about `4.23` decode tokens per second.

The model has a vocabulary of `248320` tokens. Transferring all logits after every decode step adds host and fabric work.

The packaged demo already captures decode traces with stable input buffers. It also snapshots GDN state before capture and restores it afterward.

Prefill trace capture is not part of this change. The package records an L1 conflict on the current runtime for that path.

## Goal and Success Contract

The goal is lower batch-one decode latency without a correctness regression.

A typed Nickel profile owns these values:

- Baseline receipt identity.
- Prompt corpus identity.
- Warm-up and repeat counts.
- Maximum prompt and decode lengths.
- Trace capacity and timeout bounds.
- Minimum median decode speedup.
- Maximum TTFT and tail-latency regression.
- Required token-parity and cleanup results.

A physical candidate passes only when exact greedy tokens match the eager path. The median decode rate must also exceed the profile bound.

## Architecture

A pure core owns trace eligibility, buffer-state transitions, shard-result reduction, tie handling, and evidence admission.

The imperative shell owns TT-NN tensors, trace capture, trace replay, device copies, synchronization, clocks, files, and HTTP responses.

The core uses explicit resource states:

```text
empty -> admitted -> active -> resetting -> empty
```

Every other transition fails before device dispatch.

## Decisions

### Decision: Trace decode only

**Choice:** Capture one static decode step after weights load and program warm-up. Replay it once for each generated token.

**Rationale:** Decode has stable shapes and dominates long generation. Prefill tracing has a package-matched L1 blocker.

If trace capture or replay fails validation, the service uses eager decode. The receipt names the fallback reason.

### Decision: Restore recurrent state after capture

**Choice:** Snapshot every GDN recurrent tensor before capture. Restore it before the first real decode step.

**Rationale:** Trace capture executes the graph. An uncorrected capture would advance request state before generation starts.

The service verifies state identity, shape, dtype, ownership, and generation before replay.

### Decision: Keep request buffers stable

**Choice:** Allocate bounded request slots at service start. Reuse one admitted slot for the complete request.

A slot owns token, position, RoPE, page table, KV cache, GDN state, output, and trace-binding tensors.

**Rationale:** Stable addresses support trace replay. Reuse also removes repeated allocation from request setup.

A reset clears logical state before slot reuse. Cleanup failure quarantines the slot and blocks new admission.

### Decision: Reduce vocabulary results before host transfer

**Choice:** Each device shard computes its maximum logit and global token index. The host compares only the shard winners.

**Rationale:** TP2 cannot use the current multi-device Top-K path. Greedy selection needs only one winner per shard.

Ties select the lowest global token index. Invalid shard ranges, duplicate ranges, non-finite maxima, or missing winners fail closed.

### Decision: Use matched physical A/B evidence

**Choice:** Compare eager and traced paths in order-balanced physical runs.

Both paths use the same closure, checkpoint, tokenizer, prompt corpus, mesh, generation settings, and sampling mode.

The receipt records TTFT, median decode rate, tail token latency, setup time, cleanup time, and peak allocation.

**Rationale:** Static review and simulator output cannot prove a physical speedup.

## Validation Strategy

Positive device-free tests cover valid lifecycle transitions, valid trace bindings, shard reduction, deterministic ties, and passing receipts.

Negative tests cover stale bindings, changed buffer generations, trace-capture state drift, malformed shard ranges, non-finite logits, and cleanup failure.

A bounded simulator smoke checks graph construction and state carry when the package-matched simulator supports all required operations.

Physical validation compares exact generated tokens for short, long, empty, and boundary prompts. It also checks sequential request isolation.

## Risks and Mitigations

- **Stale trace addresses:** Bind traces to slot and buffer generations. Reject every mismatch.
- **State leakage:** Reset each slot and quarantine it after any cleanup failure.
- **Hidden synchronization:** Record host gaps and TT-NN synchronization counts.
- **Argmax mismatch:** Compare complete greedy continuations with the eager path.
- **Trace capacity failure:** Retain eager decode and report the exact bounded blocker.

## Non-Goals

- Changing model equations, checkpoint bytes, tokenizer bytes, or default precision.
- Removing the eager decode path.
- Claiming better prefill performance.
- Claiming higher concurrent-request throughput.
