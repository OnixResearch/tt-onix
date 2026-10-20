# metalium-rs parity inventory

Pinned source: `brittonr/metalium-rs` at `4fbddb5d297044f8b9426e69366017b47bc55fba`
(upstream `Knight-Ops/metalium-rs` at `b22a245`).

This inventory maps the repository's two model targets to the native stack. It is a work
inventory, not a support claim.

## Native stack facts

From `docs/plans/burn-op-coverage.md` at the pinned commit:

| Trait | Native | Composed | Unsupported |
|---|---:|---:|---:|
| `FloatTensorOps` | 106 | 18 | 0 |
| `IntTensorOps` | 78 | 26 | 0 |
| `BoolTensorOps` | 37 | 2 | 0 |
| `ActivationOps` | 14 | 0 | 0 |
| `ModuleOps` | 36 | 18 | 2 (`rfft`, `irfft`) |
| `QTensorOps` | 3 | 67 | 10 (portable quantization surface) |
| `TransactionOps` | 1 | 0 | 0 |

The upstream model ladder covers ResNet-18, ViT-Tiny, and SmolLM2-135M. SmolLM2 exercises dense
attention, RoPE, grouped-query attention, and a KV cache. There is no RWKV-7, Qwen, MoE, or
gated-delta implementation.

Storage formats include F32, BF16, F16, BFP8, BFP4, and BFP2 per the mixed-storage plan. INT8 and a
general payload-preserving Tensix transpose remain open.

Mesh support splits rank-two and batched matmuls, convolution, and attention products by output
columns across cards. Chips run sequentially, each card uses one compute tile, and there is no
all-reduce or other collective.

## Gap by target

| Capability | Native status | RWKV-7 | Qwen3.6/3.8 | Blocker |
|---|---|---|---|---|
| Dense matmul / linear | Native, fast and opt-in accurate modes | Required | Required (projections, experts) | None; accurate mode costs 19–64× fast |
| Elementwise, reductions, softmax | Native | Required | Required | None |
| Conv / pool / interpolate | Native | Not central | Causal conv in GDN | Causal 1-D path not yet exercised |
| Attention + KV cache | Exercised by SmolLM2 | Not used | 10 full-attention layers | Qwen cache paging/MTP not present |
| Token shift | Composed from slice/cat | Required | Not used | None expected |
| GroupNorm / norm | Burn composition over native ops | Required (RWKV norm) | RMSNorm | Verify numerics per model |
| LoRA | Model-level matmul/add | Required | Not used | None expected |
| **WKV7 recurrent update** | **One-head SFPU kernel, ttsim gate only** | **Required** | GDN is a related but distinct recurrence | Card-0 gate, burn-tt routing, then all heads |
| **Gated DeltaNet decode** | **Missing** | Not used | Required in 30 layers | Custom kernel, state carry, precision gate |
| **MoE routing / top-k / expert dispatch** | **Missing as a fused path** | Not used | 256 routed + 1 shared expert | Routing, weight residency, BFP4/BFP8 expert formats |
| **MTP / speculative verify** | **Missing** | Not used | Required for the serving profile | Verify/commit state contract |
| **Paged / continuous KV cache** | **Missing** | Not used | Required for serving | Cache admission and eviction |
| **Mesh all-reduce / collectives** | **Missing** | Not required for 1×1 | Needed for replicated/P150x2 serving | Ethernet collective and determinism |
| Serving frontend | Out of scope upstream | Needed | Needed | Tokenizer, scheduler, OpenAI API |

## Recommended order

1. **RWKV-7 WKV7 kernel.** The host reference, composed Burn/TT gate, synthetic layer gate,
   real layer-zero gate, all-12-layer gate, logits/Python-reference comparison, and exact tokenizer
   are done, one greedy next-token step matches the harness, and the three-step stateful decode
   chain matches the harness. Prompt text now runs through the exact tokenizer and bounded greedy
   generation (`" Hi, I"`, and an eight-token continuation) with fast matmuls and batching.
   One head of the recurrence is now a fused Tensix programme (`crates/tt-kernels/src/wkv7.rs` in
   the fork): four state tiles and two vector slots in `Dst`, in-tile SFPU column folds for `sa`
   and `out`, replayed row loops for the update, and seven semaphores for the hand-offs. The ttsim
   gate `step186_wkv7` holds it to `rwkv7::step` across a carried second token (max state gap
   3.0e-8, output 7.5e-8, against a 1e-5 bound), and zeroing `a` moves the output 9.9e-2 as the
   negative control. Next: the same gate on card 0, then routing a one-head step through `burn-tt`;
   only then the layer and model. The accurate-matmul path still fragments GDDR after an 8-token
   prompt; fix that allocator limit separately. toks was
   evaluated and rejected for RWKV; see `docs/research/toks-rwkv-tokenizer.md`.
2. **RWKV-7 layer and model.** Token shift, norms, LoRA, feed-forward, and the full 24-layer
   model; compare against the existing CPU reference and the tt-metal path. Keep the service on
   tt-metal until this passes.
3. **SmolLM2 on the cards (optional bring-up).** The upstream ladder already supports it; running
   it locally gives a KV-cache/attention baseline without new kernels.
4. **Qwen GDN decode.** Port the gated-delta recurrence and its state carry. Reuse the existing
   precision candidate evidence.
5. **Qwen MoE and MTP.** Routing, expert residency, and verify/commit are separate gates. Do not
   start until the dense/GDN path is stable.
6. **Mesh collectives.** Only then consider the P150x2 serving replacement.

## Acceptance rule

A model moves only after operator coverage, state carry, numerical agreement, topology, and
physical evidence pass for that exact model revision. A package build, a simulator pass, or a
single MNIST step is not model support.

## License gate

The repository declares LGPL-2.1-only for its own code; the native stack is Apache-2.0. Keep the
package separate until the owner records a combined-work license decision before repository
binaries link the crates.
