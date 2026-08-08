## Context

`model-target-selection` defines typed profiles, an immutable target factory, architecture adapters, a common evidence envelope, and a fail-closed candidate report. Today one RWKV-7 target and one synthetic fixture use it.

The accepted spec names Qwen3.6 as the unverified candidate case. It requires reviewed upstream and machine evidence before any runnable claim.

Knight-Ops now publishes a Qwen3.6-A3B bring-up at branch `qwen3_6_a3b-bringup`. The model is `Qwen/Qwen3.6-35B-A3B`, HF architecture `Qwen3_5MoeForConditionalGeneration`. It is a Qwen3-Next-style hybrid with 30 Gated DeltaNet linear-attention layers, 10 full-attention layers, and 256 routed experts plus one shared expert. The bring-up targets one Blackhole P150 with BFP4 experts, and it reports PCC gates and firmware constraints.

The checkpoint is 26 safetensors shards of about 72 GB. Metadata files are small. Weight hashes belong to a separate bounded change.

## Decisions

### Decision: Pin the bring-up tree as the reference authority

**Choice:** Add a checked-in authority manifest. It binds the Knight-Ops URL, branch `qwen3_6_a3b-bringup`, commit `37743f58d443be1eaf70ed10eb37b5377dc56b70`, subtree `models/demos/qwen3_6_a3b`, and a BLAKE3 identity over the sparse checkout.

The `.pi/references` checkout is not in version control. The manifest is the durable pin. Admission compares the local tree against the manifest before candidate facts use it.

**Rationale:** Candidate facts need one reviewed source identity. A local-only download can drift without notice.

### Decision: Bind the HF release through metadata artifacts

**Choice:** Pin `config.json`, `generation_config.json`, tokenizer files, and `model.safetensors.index.json` at one HF revision. Each artifact gets a Nix hash and a BLAKE3 identity.

The shard index binds all 26 shard file names. Weight bytes stay unpinned in this change.

**Rationale:** Metadata files are small and reviewable. They fix architecture dimensions, tokenizer identity, and shard names without a 72 GB download.

### Decision: Extend the profile contract for hybrid state

**Choice:** Add one hybrid state kind to the Nickel contract. A hybrid profile binds KV-cache facts for attention layers and recurrent-matrix facts for linear-attention layers. Add MoE facts: expert count, experts per token, expert intermediate size, shared-expert flag, and layer-mix counts. Add a multi-shard safetensors layout and one candidate precision policy value.

Existing RWKV and fixture profiles keep their current fields and values.

**Rationale:** Qwen3.6 carries both state kinds in one model. The contract must reject a profile that binds only one of them.

### Decision: Keep the candidate profile non-runnable and metadata-only

**Choice:** Add `qwen3-6-35b-a3b-p150x1-candidate` with `runnable = false`. Its artifacts are the hashed metadata files only. Its topology is one P150 with mesh `[1, 1]` and physical device `[0]`.

The target factory refuses to expose runtime output for the profile. A later bounded change pins weight shards and simulator scope.

**Rationale:** Fail-closed selection is the accepted contract. Weight fetch and simulator coverage are bounded work on their own.

### Decision: Reuse the eleven-fact candidate core

**Choice:** Keep the `CandidateFacts` schema unchanged. Author one checked-in Qwen3.6 facts JSON with all eleven reviewed facts. Encode the firmware floor, pinned tt-metal python_env, fork base identity, memory budget, and ttsim operator-coverage audit inside the reviewed fact text.

Produce the verdict with the existing report binary and check in the deterministic receipt.

**Rationale:** The core gate is completeness. Qwen-specific semantics live in reviewed fact text and fixtures, not in new gate code.

### Decision: Scope the verdict to one bounded simulator change

**Choice:** A complete fact set makes the candidate eligible for one bounded simulator change. The report does not claim that ttsim covers every required operator. The operator audit names each uncovered or unknown operator.

If the simulator path needs an unmatched TT-Metal branch or unsupported feature, the lifecycle rule records the mismatch and permits one bounded hardware attempt in a later change.

**Rationale:** The accepted workflow advances simulator smoke work before hardware. A useful negative verdict is a valid result.

## Validation Strategy

Device-free positive checks cover the hybrid contract export, the candidate profile evaluation, and a complete Qwen3.6 report verdict.

Negative checks cover a drifted reference identity, a hybrid profile with one missing state component, a runnable Qwen profile, an unhashed weight artifact, and an unreviewed candidate fact.

Existing RWKV target, synthetic fixture, profile freshness, and report self-test checks must keep their current results.

No validation in this change can prove simulator coverage, hardware capacity, execution, or performance.

## Risks / Trade-offs

- The fork can drift from the pinned commit. The authority manifest and admission check fail on content drift.
- The HF release can move or vanish. Metadata artifact hashes fail closed on revision drift.
- The bring-up names the HF architecture `Qwen3_5MoeForConditionalGeneration` while the product name is Qwen3.6. The facts record this naming mismatch as reviewed text.
- ttsim operator coverage is unknown for several required operators. The operator audit records each gap instead of assuming coverage.
- Fact strings carry rich reviewed content. Fixtures pin the exact reviewed text so drift is visible.

## Non-Goals

- Download or hash the 72 GB weight shards.
- Run ttsim, open a device, or measure performance.
- Implement Qwen operators, kernels, or device adapters.
- Support serving, batching, sampling, or vLLM integration.
- Claim Qwen3.6 availability, compatibility, or support.
