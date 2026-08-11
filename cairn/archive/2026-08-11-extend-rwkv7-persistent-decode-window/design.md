## Context

The production profile admits a two-step Metal Trace. Trace capture and normal reference execution occur before timing. One bounded replay carries selected tokens and FP32 recurrent state across both steps.

The trace implementation already accepts a step count, but capability validation binds every request to the profile maximum. The lifecycle core also uses the profile maximum as the request completion bound.

## Decisions

### Decision: Keep one immutable base profile and derive a bounded request profile

**Choice:** The Nickel profile owns a closed, sorted set of admitted window lengths and one maximum bound. A pure core function accepts only a listed window and derives request-local step, output, and cancellation bounds.

**Rationale:** This preserves one BLAKE3 profile identity while it lets two-token and longer requests use the same reviewed runtime. Unsupported windows fail before trace capture.

### Decision: Retain one replay per complete window

**Choice:** Capture every selected window as one finite Metal Trace. Do not add an unbounded device loop.

**Rationale:** One replay preserves zero per-token host submissions inside the window. A finite profile-owned window also preserves bounded host recovery.

### Decision: Use the retained conversational prompt oracle

**Choice:** Use the existing `Hello` chat prompt tokenization and pinned FP32 CPU-reference outputs as the longer-window correctness oracle.

**Rationale:** This replaces the synthetic admission prompt with a repository-owned, tokenizer-backed prompt while preserving a deterministic reference.

### Decision: Keep admission per window

**Choice:** Each production window needs simulator, exact parity, profiler, and order-balanced physical evidence. A failed longer window does not remove the accepted shorter window.

**Rationale:** A profile member is a separate performance and trace-capacity claim. Admission cannot generalize from a shorter trace.

## Risks / Trade-offs

- Trace storage and retained tensor lifetimes grow with the selected window.
- A replay cannot observe host cancellation while it is executing. Cancellation is checked before replay, and the finite window bounds the next host observation point.
- A longer trace can preserve parity but lose its physical speedup. That result keeps the window unadmitted.
- Simulator evidence does not prove hardware performance or determinism.
