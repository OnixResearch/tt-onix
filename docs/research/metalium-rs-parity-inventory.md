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
| **WKV7 recurrent update** | **One-head SFPU kernel; all 12 heads in one job list across 12 tiles, the layer, the 12-layer model, the decode chain and fixed-prompt generation run on it at card 0** | **Required** | GDN is a related but distinct recurrence | Head-batched kernel; the 768x768 projections |
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
   and `out`, row loops for the update, and seven semaphores for the hand-offs. Row vectors arrive
   raw (`[1, 32]` images) and the unpacker lays them into the row-broadcast form; `v` is a `[64, 1]`
   column the mover broadcasts. Gate `step186_wkv7` holds it to `rwkv7::step` across a carried second
   token on ttsim and card 0 (both: max state gap 3.0e-8, output 7.5e-8, against a 1e-5 bound; card 0
   run `1792472083`, 0.40 s), and zeroing `a` moves the output 9.9e-2 as the negative control. The
   GDDR-resident form (`wkv7::work`, `Session::wkv7_step`) and its `burn-tt` route
   (`burn_tt::wkv7_step`, `rwkv7::step_head_device_attached`) pass the same gate on ttsim and card 0
   (`step187_wkv7_burn`, run `1792472975`, 0.91 s). All twelve heads run as twelve launches
   (`rwkv7::step_device_fused_attached`; card 0 run `1792505756`, 0.36 s, state 4.5e-8, output
   1.5e-7; `wkv7-gate --card 0 --fused` passes at 1.5e-7/4.5e-8 and 8.9e-8/2.2e-8 carried). Next:
   the layer and model now run on the fused recurrence: the layer's Goose form is the same update
   with `a = -key_key`, `b = key_key * adaptation`, `w = decay`. `layer_step_fused` is opt-in
   (`--fused` on the layer and model gates). Card 0, pinned checkpoint: layer gate in accurate mode
   token1 hidden 3.6e-7 / token2 2.4e-7 / states 7.5e-8 and 1.7e-7; the 12-layer model gate passes
   with final hidden 1.671e-3 (relative 2.245e-5), logits 2.708e-4 (relative 1.377e-5), recurrent
   state 4.168e-4 (relative 7.577e-6), greedy token 2, about 19 s. `step188_wkv7_layer` runs a layer
   on ttsim and card 0 (default matmul mode, 4.95e-4 -- TF32 rounding, not the kernel). The generate
   gate's `--fused` path passes on card 0: the three-step decode chain picks token 1 at every step,
   and the fixed prompt generates `" Hi, I"` (ids `[3880, 45, 308]`, equal to Flex). Cost work:
   `w`/`b` are shared by both row tiles (18 vector unpacks a head, down from 22), and the twelve
   heads are twelve jobs in one `Work` (`burn_tt::wkv7_step_batched`), one submission with one
   state/output slice rather than twelve `cat`s. The layer gate's run falls from about 3.3 s to 2 s;
   the model gate stays about 19 s on one tile. The gates take `--tiles N`, so the head jobs and
   every other op spread over the session's units: card 0 `--fused` drops the layer gate from 1.56 s
   to 0.84 s on twelve tiles, the 12-layer model gate from 18.47 s to 6.45 s, and fixed-prompt
   generation from 10.93 s to 2.83 s, with the same gaps, token and text. `step187` also gates the
   twelve-head step on twelve tiles (card 0, 0.51 s for one token). Next: a head-batched kernel and
   the 768x768 projections. The accurate-matmul path still fragments GDDR after an 8-token
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
