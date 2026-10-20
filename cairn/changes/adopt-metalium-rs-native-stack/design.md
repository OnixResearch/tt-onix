## Context

`brittonr/metalium-rs` at commit `dd5d316e3de31150fcee935ff321a6fc8ac863c2` is a fork of `Knight-Ops/metalium-rs` at `b22a245`. The fork adds `LICENSE` (Apache-2.0), fixes one test assertion, and adds `crates/rwkv7`: the WKV7 host reference, its composed device gate, a synthetic full-layer Flex/Tt differential gate, a real-checkpoint layer-zero loader, the all-12-layer gate, logits/oracle comparison, the exact tokenizer, the greedy next-token check, the stateful decode chain, and bounded prompt generation.

The full fork validation tier passes: 1,540 tests, 0 failures, 46 ignored, 259 binaries. The Nix package builds the shipped crates and `tt-mnist`.

The repository's Rust code is host-side validation and evidence today. Device work still runs through TT-Metal/ttnn, llama.cpp Metalium, or the ttWKV7 runtime. `metalium-rs` has no RWKV-7, Qwen, MoE, or gated-delta implementation, so no model service can switch yet.

## Decisions

### Decision: The fork is the package authority; upstream remains the comparison base

**Choice:** Pin the fork commit and record the exact delta from upstream. Keep upstream as the reference for behavior the fork did not change.

Update the fork by rebasing the reviewed changes onto a reviewed upstream commit. Do not merge unrelated upstream work into the pinned package without a new validation run.

**Rationale:** The fork needs a license file and the test fix to pass its own tier. A recorded delta keeps future upstream refreshes auditable.

### Decision: Pin the Rust toolchain, not the ambient rustup install

**Choice:** Add `rust-overlay` and pin Rust 1.97.1 with `riscv32im-unknown-none-elf` and `llvm-tools`.

The package uses `makeRustPlatform` with that toolchain, so `rustc`, `cargo`, the firmware target, and `llvm-objcopy`/`llvm-objdump` all resolve from Nix inputs. Do not reuse the host rustup installation.

**Rationale:** The nixpkgs development shell lacks the firmware target and LLVM tools. An ambient rustup toolchain makes the package irreproducible.

### Decision: Build the shipped stack and the MNIST binary

**Choice:** Build the workspace with only dev-only crates excluded. Provide the pinned MNIST corpus and install `bin/tt-mnist`.

The excluded crates need the ISA specification tree, ttsim payloads, or model weights. Keep them in the source validation lane, not the package.

**Rationale:** An empty library-only derivation proves compilation but gives no runnable artifact. `tt-mnist` is self-contained and exercises firmware, kernels, device transport, and the Burn backend.

### Decision: Start WKV7 as a composed gate, then fuse

**Choice:** Add `crates/rwkv7` with the pure F32 host reference and the same equation composed from
Burn tensor operations on `TtBackend`. Use `accurate_matmul` and verify two-token state carry on
card 0.

Do not call this a fused kernel or a model. The fused Tensix recurrence remains the next gate.

**Rationale:** The composed path proves the equation, the state contract, and the existing backend
ops before a new kernel exists. It gives a reference the fused kernel can be compared against.

### Decision: Prove the layer graph with synthetic weights before real weights

**Choice:** Implement one layer from the pinned equations and compare `Flex` FP32 with `TtBackend`
on identical synthetic weights. Use layer zero, which exercises the pre-norm, token shift, time
mixing LoRA, WKV recurrence, group norm, correction, output projection, and squared-ReLU FFN.

Do not call this a real-weight layer or a model. Real checkpoint loading and nonzero-layer v-LoRA
remain next.

**Rationale:** The synthetic gate isolates backend disagreement from checkpoint layout and weight
conversion. It finds missing ops and shape bugs before a 382 MB checkpoint is involved.

### Decision: Load the real checkpoint before nonzero layers

**Choice:** Read the pinned BF16 safetensors checkpoint for layer zero, convert to F32, and compare
`Flex` with `TtBackend`. The loader handles the checkpoint's rank-128 g-LoRA, per-head `r_k`, and
the 3-D token-shift factors. Token inputs are embedding rows 1 and 2.

Do not extend to nonzero layers until the v-LoRA path is implemented and checked.

**Rationale:** Real weights expose layout and shape errors that synthetic weights cannot. Layer
zero is the smallest real-weight slice with no cross-layer value anchor.

### Decision: Run all 12 layers before logits

**Choice:** Carry the layer-zero value anchor through all 12 real-weight layers and compare the
final normalized hidden vector and every recurrent matrix between Flex and TtBackend.

Do not evaluate the 65536x768 logits in this gate. Keep the comparison relative to each output's
max magnitude.

**Rationale:** The 12-layer path exercises the nonzero-layer v-LoRA and state carry. The logits add
a large matmul without adding recurrence coverage, and the Python-reference comparison is a
separate oracle.

### Decision: Compare logits against the harness's Python equation reference

**Choice:** Evaluate `lm_head`, write the reference `final_hidden`, `logits`, and `states` from the
harness's Python equation source, and compare both Flex and TtBackend against those files.

Use the harness's 0.5 final-hidden, 1.0 logit, and 1.0 recurrent-state bounds for the oracle. Do
not claim tokenization, generation, or serving parity from this comparison.

**Rationale:** The Python reference is the independent equation oracle. It caught no layout error
at this point, which is evidence the Rust port and checkpoint mapping agree.

### Decision: Implement the RWKV tokenizer directly; toks cannot load it

**Choice:** Implement the vocabulary's leftmost-longest byte trie in `crates/rwkv7`. Pin
Python-derived vectors, including emoji and `U+10FFFF`.

`toks` 0.3.2 was evaluated and refused the model directory: it needs a `tokenizer.json` or tiktoken
model. RWKV ships neither, and 486 vocabulary tokens are not valid UTF-8, so a JSON added token
cannot represent them.

**Rationale:** The tokenizer is small and exact as a byte trie. A general tokenizer library cannot
reproduce raw-byte leftmost-longest matching through its JSON model surface.

### Decision: Check one greedy token before a generation loop

**Choice:** Rank the top two logits and compare them with the harness token checkpoint: token 2,
logit 2.8641083; runner-up 33, logit 0.89640886; margin 1.9676995.

Do not call this a generation loop, sampler, or serving result.

**Rationale:** The top-two check catches final-norm, head, and ranking errors with one more scalar
comparison. A generation loop adds state transitions and stop conditions that need their own gate.

### Decision: Carry state across a bounded decode chain

**Choice:** Start from seed token 1 with zero state and run three greedy steps, carrying every
layer's matrix and token-shift vectors. Compare generated ids and logits with the harness decode
checkpoint.

Keep the chain bounded and fixed. A prompt-generation endpoint, sampler, and stop policy remain
separate work.

**Rationale:** One greedy step does not exercise state transitions between generated tokens. The
three-step chain is the smallest gate that does.

### Decision: Generate from text only after wrapper tokenization

**Choice:** Add wrapper tokenization (special id 0, wrapper EOS 65530) and a bounded greedy
generation from the fixed chat prompt. Expect `" Hi, I"`.

The prompt app sets `TT_BATCH=0`: the batched path exhausts contiguous GDDR slots after the
8-token prompt. The default pipeline is fine. Treat that as an allocator limitation to fix, not a numerical difference.

**Rationale:** Text generation is the first end-to-end path from bytes to the model and back.
Running it unbatched keeps the migration gate honest while the allocator behavior is unresolved.

### Decision: Reset cards between TT-Metal and native execution

**Choice:** Run `tt-smi -r` after stopping a TT-Metal service and before native-stack execution.

Without it, the first native process failed at `element-wise 0x2 ... queued list 2 failed`; a
second process passed, and a reset made the first run deterministic.

**Rationale:** The native stack must not depend on a failed first process to clear the card. The
reset is an operational gate, not a source workaround.

### Decision: Migrate by parity gates, not by service replacement

**Choice:** Route new Rust device, firmware, kernel, and Burn-backend work to `metalium-rs`.

Keep each existing model service on its current runtime until the native stack passes that model's gates: operator coverage, recurrent-state carry, numerical agreement, mesh topology, and physical evidence.

**Rationale:** The native stack has no model parity today. Replacing a serving path on build success would convert partial evidence into a support claim.

### Decision: Keep the package separate until license compatibility is decided

**Choice:** Distribute `metalium-rs` as a separate Nix package. Do not link its crates into
repository-authored binaries until the owner records a license decision.

The repository declares LGPL-2.1-only for its own code. The fork is Apache-2.0. A package
derivation aggregates the work; a linked binary is a combined work with different obligations.
The package build and physical smoke do not settle that question.

**Rationale:** License scope is a distribution fact, not an engineering inference. The native
stack can be evaluated and run without pretending the combined-work terms are resolved.

### Decision: Supersede the cxx bridge for native device work

**Choice:** Do not implement `package-libtt-metal-cxx`. Keep its proposal and design as a reviewed record. `metalium-rs` does not expose TT-Metal host APIs, so the two are not interchangeable.

**Rationale:** The migration direction is native Rust. Maintaining a second Rust-to-C++ bridge adds a toolchain and ABI surface the native stack removes.

## Validation Strategy

Positive checks cover the fork pin, the license file, the F16 test fix, the WKV7 host reference, the full validation tier, a hermetic package build, the installed `tt-mnist`, `wkv7-gate`, `rwkv7-layer-gate`, and `rwkv7-model-gate` binaries, the WKV7 two-token state-carry gate, the synthetic full-layer Flex/Tt gate, the real-checkpoint layer-zero gate, and the all-12-layer gate on card 0.

Negative checks cover source drift, an unpinned toolchain, a missing firmware target, a missing license, and a package that claims a model runtime. The installed output excludes dev-only crates; the check phase runs their simulator-backed tier, and no check opens hardware.

Physical validation is separate. The first hardware gate is `tt-mnist --steps 1 --card 0` on one card; mesh work follows only after the single-card path is measured.

## Risks / Trade-offs

- The fork can drift from upstream. The recorded delta and a revalidation requirement keep it reviewable.
- The composed WKV7, synthetic layer, real layer-zero, and all-12-layer gates are correctness evidence only; the logits/oracle comparison is equation-level, not tokenization, generation, fused-kernel cost, or performance.
- Native execution after a TT-Metal service needs a board reset; omitting it makes the first process fail.
- Rust 1.97.1 is pinned through `rust-overlay`; a future lock update can change the overlay revision even if the toolchain version stays fixed.
- `tt-mnist` embeds about 11 MB of compressed data and adds build time. It is the only self-contained native binary today.
- Excluding dev-only crates leaves the conformance and simulator gates outside the package check.
- Model migration can stall on missing kernels. That is an explicit blocker, not a packaging defect.
- LGPL-2.1-only versus Apache-2.0 combined-work terms remain unresolved; keep the package separate
  until the owner decides.

## Non-Goals

- Claim Qwen3.6/3.8, RWKV-7, MoE, or gated-delta support on `metalium-rs`.
- Run hardware or simulator workloads in the default flake check.
- Remove TT-Metal, TT-NN, ttsim, or the current model runtimes.
- Track upstream automatically or publish the fork.
- Implement the cxx bridge.
