## Context

The existing recipe already uses BF4 MLP gate/up. The retained BF16 MTP head restores acceptance that the FP8 head lost. The Adartras bundle is a MoE reference, not a dense TP2 accuracy oracle.

## Decisions

The first screen changes MLP down projections only. Candidate recipes identify layer sets, weight format, accumulation type, and math fidelity. Keep FP32 recurrence and BF16 MTP weights fixed.

Each recipe gets a separate BLAKE3-bound cache namespace and a conversion manifest. Measure actual tensorbin and device allocation sizes, including exponent and padding overhead. Do not infer TT BFP8 from an HF FP8 file name.

Use existing per-module reference tests and distribution-admission cores. Add teacher-forced decode and chunk-invariant coverage rather than a new monolithic benchmark tool.

Evaluate BF8 KV separately under the current context bound. Any larger context depends on a separately admitted runtime and memory profile. Head or GDN projection precision changes come only after the lower-risk recipe has a measured result.

## Gates

Profiles name the layer set, prompt corpus, teacher-forced steps, chunk boundaries, finite-value checks, task suite, and numerical limits. Freeze the profile before a hardware screen.

Positive fixtures accept complete recipes and valid comparison rows. Negative fixtures reject stale caches, mixed scales, wrong dtypes, missing layers, non-finite logits, state drift, and unsupported sampled-versus-greedy claims.

Physical evidence includes task quality, MTP acceptance for shared-head changes, memory, decode rate, and request latency. A sampled distributional pass cannot admit a changed greedy route.

## Alternatives and rollback

Blanket BF4 and FP8 MTP-head conversion are rejected as defaults. A prefill-only evaluation cannot establish decode quality. A tiny operator PCC test cannot establish long state carry.

Keep the current recipe and caches unchanged. Owner approval of a measured quality trade-off is explicit and separate from this research proposal.
