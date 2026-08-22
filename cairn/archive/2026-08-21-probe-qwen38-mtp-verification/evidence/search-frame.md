# Search frame

## Question

Can the pinned P150x2 runtime provide a correct and economically useful target-verification base before the checkpoint MTP head is loaded?

## Competing routes

- Fuse the recurrent GDN graph for single-token decode.
- Tune common MLP weight delivery and matmuls.
- Use the existing masked prefill path as a teacher-forced verifier.
- Build a native short-width hybrid verifier, then add the MTP drafter.
- Use continuous batching for aggregate throughput instead of single-stream latency.

## Discriminating evidence

The active tensor cache shows that common MLP weights form the largest estimated decode weight bucket. This makes a GDN-only optimization an incomplete target.

The checkpoint has one MTP layer, but the adopted mapper removes every `mtp.*` weight. Loading those weights before a target verifier exists would not establish a usable path.

The physical widths `1`, `2`, and `4` probe tested the only available multi-token target path in runtime 0.74. The path failed target tokens, GDN state, stability at width one, and execution-time amortization.

## Decision

Reject the masked fallback as an MTP verifier. Do not select a GDN-only fusion from static evidence. The next MTP prerequisite is a native short-width hybrid verifier with exact carried-state semantics.

If that verifier cannot beat traced sequential execution, return to direct P150x2 component timing. Compare MLP, GDN, attention, and LM-head costs before selecting a kernel target.
