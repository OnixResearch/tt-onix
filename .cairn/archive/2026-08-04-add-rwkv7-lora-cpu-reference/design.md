# Design: RWKV-7 LoRA CPU reference

## Context

The current harness pins `RWKV/RWKV7-Goose-World2.8-0.1B-HF` at revision `d81965cb4e1a9f96696b4f70b84212b8f2e43216`. The checkpoint has 12 layers, hidden width 768, 12 heads, and head width 64.

The current CPU equation reference runs inference under `torch.no_grad()`. It validates decode behavior but does not define training gradients.

Appendix H of the RWKV-7 paper provides a naive PyTorch WKV7 forward and backward reference. The upstream `train_temp` code provides the current training parameter structure and sequence behavior.

## Decisions

### Decision: Keep the first model boundary exact

**Choice:** Support only the pinned World2.8 0.1B checkpoint and its pinned tokenizer artifacts in this change.

**Rationale:** The repository already has strong provenance for this checkpoint. A single exact model reduces compiler and gradient uncertainty in later Cairns.

### Decision: Use LoRA, not full-parameter fine-tuning

**Choice:** Freeze every base checkpoint tensor. Add trainable LoRA matrices only to an explicit role-based allowlist.

The default allowlist covers attention receptance, key, value, and output projections. It also covers feed-forward key and value projections.

The language-model head, embeddings, normalization parameters, time-mix parameters, and existing RWKV low-rank parameters remain frozen by default.

**Rationale:** The 0.1B base model fits one P150A. LoRA lowers optimizer memory and limits the first gradient surface.

An `all-linear` selector is not allowed. It can silently include the language-model head or future modules after source changes.

### Decision: Use a pure sequence equation for the reference

**Choice:** Implement the full RWKV-7 sequence path from tensor inputs and explicit state. Use the paper Appendix H recurrence for bounded WKV7 forward and backward fixtures.

**Rationale:** The existing ttWKV7 kernel targets token decode. Training needs sequence outputs and gradients across time.

The reference uses FP32 arithmetic after it loads BF16 checkpoint values. This precision class matches the existing CPU harness boundary.

### Decision: Validate gradients through two mechanisms

**Choice:** Compare PyTorch autograd with the paper-matched manual WKV7 backward equation. Also use bounded finite differences for selected scalar elements.

**Rationale:** Autograd alone can repeat a wrong forward equation. The manual backward path and finite differences add distinct failure detection.

Every tolerance has a named Nickel configuration value. No tolerance remains an unexplained source literal.

### Decision: Define a typed local SFT data contract

**Choice:** Accept local JSONL records with nonempty `prompt` and `response` strings. Tokenize with the pinned RWKV World tokenizer.

The label sequence masks prompt tokens and padding. It includes response tokens under the pinned special-token policy.

**Rationale:** Prompt masking gives a clear supervised fine-tuning objective. Local data keeps network access outside the training process.

Malformed records, empty target labels, overlong samples, unknown fields, and tokenizer drift fail before model execution.

### Decision: Author experiment configuration in Nickel

**Choice:** Define contracts for checkpoint identity, tokenizer identity, data paths, static sequence profile, LoRA rank and scale, target roles, optimizer, learning rate, gradient accumulation, clipping, seed, output path, and checkpoint policy.

**Rationale:** One typed configuration can feed the CPU and TT-XLA runners. Cross-field rules prevent invalid target sets and resume mismatches.

### Decision: Separate deterministic cores from file and framework shells

**Choice:** Keep record admission, token-label planning, LoRA target selection, parameter-freeze planning, checkpoint manifests, and receipt validation pure.

The shell loads files, constructs PyTorch modules, runs tensor operations, and writes output artifacts. Random values come from an explicit seeded generator.

**Rationale:** Most policy can run in device-free unit tests without filesystem or framework mocks.

### Decision: Save adapters as portable Safetensors

**Choice:** Save only LoRA tensors and required metadata. Bind the base checkpoint, tokenizer, configuration, data manifest, optimizer policy, step, and tensor artifact with BLAKE3.

**Rationale:** Adapter-only artifacts are small and cannot silently replace frozen base weights.

Resume rejects changed base, tokenizer, targets, shapes, optimizer policy, or data identity.

## Test Design

Positive tests cover data admission, prompt masking, target selection, initial LoRA equivalence, forward values, manual gradients, autograd gradients, finite differences, one optimizer step, adapter-only mutation, deterministic reruns, save, load, and resume.

Negative tests cover malformed JSONL, empty prompts, empty responses, no active labels, stale model or tokenizer identities, unknown targets, duplicate targets, nonfinite tensors, shape mismatch, base-weight mutation, invalid rank, invalid clipping, corrupted adapters, and incompatible resume state.

A tiny fixed corpus supplies positive training evidence. A changed-label corpus and malformed records supply negative evidence.

## Risks and Trade-offs

- The naive recurrence is slow. Fixtures use bounded shapes and do not claim production speed.
- LoRA target policy affects adaptation quality. The first slice values a narrow and auditable surface.
- FP32 CPU values can differ from TT-XLA BF16 values. Later Cairns use named tolerances and compare semantic invariants.
- One tiny corpus cannot prove useful generalization. The receipt states this limit.

## Claim Boundary

A passing CPU receipt proves the bounded reference equations, selected LoRA plan, deterministic SFT labels, adapter-only update, and artifact integrity. It does not prove Tenstorrent support, useful convergence, dataset quality, or production fine-tuning quality.

## Authority Sources

- `.pi/references/rwkv-7/rwkv-7-goose-arxiv-2503.14456v2.txt`, especially Appendix H.
- `.pi/references/rwkv-7/RWKV-LM/RWKV-v7/train_temp/README.md`.
- `.pi/references/rwkv-7/RWKV-LM/RWKV-v7/train_temp/rwkv7_train_simplified.py`.
- `pkgs/rwkv-layer-harness/reference/rwkv7_torch_equation_reference.py`.
