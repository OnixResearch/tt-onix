## Context

The candidate revision is `9f9cd4fd590f4b606bd0981a4fe0b6403eb38ec9`, the resolved 0.77 tag. The September report cites its fused GDN binding. The current model source is a separately pinned adoption with local TP2 patches.

## Decisions

Create a separate Nix candidate package. Do not replace the repository-wide TT-Metal input. Bind every model, runtime, compiler, driver, Python, and simulator component. Nix generates lock changes through its normal commands.

First compare the adopted model against the candidate API. Retain Qwen3.8 registration, layer-type authority, and TP2 shape checks. Prefer the smallest compatible model adaptation over a wholesale source replacement.

The capability matrix names fused GDN, grouped valid-length masks, stable state copies, paged cache operations, fused SwiGLU AGMM, and TP2 output reduction. An unavailable capability keeps its fallback.

The fused GDN adapter uses the model-selected chunk size, FP32 state, explicit Q/K normalization, and device-owned constants. The current masked path constructs host masks. A traced masked candidate must stage those masks as stable inputs rather than upload them during capture.

## Validation sequence

The device-free stage builds the candidate and validates signatures, exports, model identity, and local kernel compilation. Positive and negative fixtures cover supported shapes and missing or mismatched capabilities.

The simulator stage opens the expected architecture and runs a representative GDN operation with state carry. Unsupported trace or fabric behavior produces a bounded blocker, not simulated admission.

An approved physical stage compares the same prompts and weights on control and candidate. It covers full chunks, masked tails, second requests, nonzero recurrent state, B4, and near-boundary context. It measures memory, TTFT, decode, and complete request latency.

## Alternatives and rollback

Backporting the fused operator alone remains an alternative only after the dependency diff is known. A version-number upgrade without a closure manifest is rejected.

Promotion is per capability. The prior runtime and every required fallback remain available. Context expansion and production deployment require separate evidence and owner approval.
