# Design: RWKV-7 LoRA bring-up on one P150A

## Context

The platform Cairn supplies an immutable TT-XLA runtime and a validated dual-P150A system. The CPU Cairn supplies exact model, data, LoRA, equation, gradient, optimizer, and checkpoint authority.

TT-Blacksmith has LoRA examples for transformer models on P150. It does not list RWKV-7 support.

RWKV-7 training needs a sequence WKV7 forward and backward path. The current repository ttWKV7 operator targets inference and does not provide this training contract.

## Completion Contract

A valid result has all of these facts:

- The pinned full model compiles for one admitted static sequence profile.
- The full forward, loss, backward, clipping, and adapter optimizer graph runs on TT.
- No unsupported model or optimizer operation uses silent host fallback.
- Logits, loss, selected activations, gradients, and adapter deltas satisfy named CPU comparison tolerances.
- At least one adapter changes and every base tensor stays unchanged.
- Save, load, and resume preserve the admitted artifact contract.
- Device ownership and health are restored after success or failure.

A finite loss alone is not completion evidence.

## Approach Registry

| Family | Mechanism | State | Discriminating evidence |
|---|---|---|---|
| TT-XLA direct | Trace a pure fixed-shape PyTorch recurrence into StableHLO | Selected first | Focused WKV7 forward and backward compile receipt |
| TT-XLA composite | Add an explicit compiler composite or custom operation | Blocked pending direct probe | Exact unsupported operation and lowering boundary |
| Native TT-NN training | Build native WKV7 forward, backward, optimizer, and framework integration | Deferred | A separate design with a smaller verified gap |
| Hybrid host recurrence | Run WKV7 on the CPU and projections on TT | Rejected | It cannot satisfy full graph placement |

The selected route does not assume compiler support. A failed direct probe is a bounded blocker, not permission for host fallback.

## Decisions

### Decision: Start with direct fixed-shape TT-XLA lowering

**Choice:** Express WKV7 with ordinary PyTorch tensor operations and a static sequence loop. Compile focused forward and backward probes before full model compilation.

**Rationale:** This route reuses the CPU equation and TT-XLA autograd path. It avoids a new native kernel until evidence shows that one is necessary.

The sequence profile is named in Nickel. It owns micro-batch size, sequence length, accumulation count, dtypes, and tolerance identities.

### Decision: Stop on an unsupported WKV7 lowering

**Choice:** If direct lowering fails its admitted compile, memory, placement, or numerical contract, emit an unsupported-operation receipt and stop this change.

**Rationale:** A custom compiler operation changes the architecture and verification surface. It needs a separate reviewed decision based on the exact blocker.

### Decision: Use one logical device without hiding the second card

**Choice:** Pass all host Tenstorrent devices to the runtime. Select one logical `1x1` submesh through the supported TT-XLA device API.

**Rationale:** Upstream states that container device isolation is not supported. Logical selection preserves that rule and gives a single-device numerical baseline.

The runtime must not open the unused logical device for model work. Platform health evidence still covers both physical cards.

### Decision: Keep base tensors frozen and resident

**Choice:** Load the pinned base checkpoint once, place frozen base tensors on the selected device, and update only admitted LoRA tensors.

**Rationale:** Host-device transfer inside each step can hide placement faults and reduce performance. Frozen resident weights give a clearer training graph.

### Decision: Compare semantic checkpoints across precision classes

**Choice:** Compare these values with named tolerances:

- Selected logits and complete loss.
- WKV7 output and final state at selected layers.
- Selected LoRA gradients before clipping.
- Gradient norm and clipping result.
- Selected adapter deltas after the optimizer step.

**Rationale:** Full byte equality is not valid across FP32 CPU and BF16 device arithmetic. Named numerical profiles preserve explicit limits.

The comparison also validates finite values, shapes, active label count, trainable parameter count, and base-weight identities.

### Decision: Reject silent host fallback

**Choice:** Save compiler and placement evidence for the complete trainable graph. Reject any unsupported fallback for model forward, loss, backward, clipping, or optimizer work.

**Rationale:** A passing script can otherwise execute critical work on the host and overstate accelerator support.

Host preprocessing, artifact writes, receipt generation, and orchestration remain allowed shell work.

### Decision: Use acceptance profiles, not source literals

**Choice:** Store static shapes, compile limits, memory limits, numerical tolerances, minimum run steps, loss criteria, health limits, and timeouts in typed Nickel profiles.

**Rationale:** Named profiles are reviewable and avoid unexplained numeric literals in code.

### Decision: Reuse adapter artifacts from the CPU contract

**Choice:** Use the same Safetensors adapter names and BLAKE3 manifest roles. Add TT-XLA graph, placement, device, and precision evidence.

**Rationale:** One portable adapter format simplifies CPU comparison and later dual-card resume.

## Test Design

Focused positive probes compile and run WKV7 forward, WKV7 backward, LoRA projections, masked loss, gradient norm, clipping, and adapter update.

Focused negative probes include invalid static shapes, unsupported dynamic shapes, nonfinite values, missing labels, disconnected adapters, forbidden fallback, stale compiler artifacts, and tolerance violations.

The one-step hardware run compares the exact same global batch with the CPU reference. The bounded multi-step run uses a fixed tiny corpus and requires a valid loss-improvement criterion.

Resume validation compares an uninterrupted path with a save-and-resume path under the same named profile.

## Adversarial Audit

The validation must try to falsify these common false completions:

- Only the WKV7 microkernel ran on TT.
- Loss ran on the host.
- Adapters received no gradients.
- The optimizer changed a base tensor.
- A stale compile artifact supplied placement evidence.
- A finite but wrong gradient passed.
- The second card changed the selected single-device result.
- Cleanup reported success after a close failure.

## Risks and Trade-offs

- A static loop can create a large StableHLO graph. The compile probe owns an explicit graph-size and compile-limit policy.
- BF16 gradient noise can need wider tolerances than inference. Tolerances remain named and tensor-specific.
- The 0.1B model can underuse a P150A. Correctness comes before utilization in this slice.
- TT-XLA source changes can alter placement reports. The runtime manifest binds the parser and schema.

## Claim Boundary

A passing receipt proves bounded RWKV-7 LoRA training behavior on one logical P150A under one exact runtime and profile. It does not prove two-card scaling, long-context support, useful generalization, speedup, or other RWKV models.
