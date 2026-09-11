# Qwen3.6-A3B candidate target

The repository contains one non-runnable Qwen3.6 candidate profile.

The target name is `qwen3-6-35b-a3b-p150x1-candidate`.

This profile records reviewed facts. It does not provide a runtime, command, server, or hardware package.

## Authority

`model-targets/qwen3-6/authority.ncl` is the reviewed authority manifest.

It pins these upstream source facts:

- Repository: `https://github.com/Knight-Ops/tt-metal.git`
- Branch: `qwen3_6_a3b-bringup`
- Commit: `37743f58d443be1eaf70ed10eb37b5377dc56b70`
- tt-metal base: `3218270556c5ff58eba1e68d112d09a11b37dc0e`
- Subtree: `models/demos/qwen3_6_a3b`
- Subtree BLAKE3: `4f4b267a13b29b55ce74a6182c2e1182dfaa7496c0ae6da8f889a0ea3ed5b87e`

The manifest also pins the Hugging Face release:

- Model: `Qwen/Qwen3.6-35B-A3B`
- Revision: `995ad96eacd98c81ed38be0c5b274b04031597b0`
- License: `Apache-2.0`
- Architecture: `Qwen3_5MoeForConditionalGeneration`
- Text model type: `qwen3_5_moe_text`
- Release config Transformers version: `4.57.1`

Qwen3.6 uses Qwen3.5 architecture names in its release metadata. The profile preserves this upstream naming fact.

## Metadata boundary

The profile pins nine small metadata artifacts.

These files include model configuration, generation configuration, tokenizer data, the license, and the checkpoint index.

The checkpoint index describes 1,045 tensors in 26 shards. It reports 71,903,645,408 checkpoint bytes.

The authority manifest records all 26 shard names. It also checks that the names are unique.

The profile does not fetch or hash the 26 checkpoint shards. No checkpoint weight bytes are in the repository.

## Hybrid architecture

The text model has 40 layers.

- 30 layers use Gated DeltaNet linear attention.
- 10 layers use full attention.
- Full attention uses 16 query heads and 2 key/value heads.
- Full-attention head size is 256.
- Linear attention uses 16 key heads and 32 value heads.
- Linear-attention head size is 128.
- The causal convolution kernel size is 4.
- The MoE has 256 routed experts and selects 8 experts per token.
- The model also has one shared expert.

The profile state kind is `hybrid_kv_recurrent`.

Full-attention layers carry fixed-shape key and value caches. Linear-attention layers carry recurrent matrices and causal-convolution history.

## Candidate precision and topology

The candidate precision is `bf4_experts_bf8_attention_bf16_activations_candidate`.

The candidate topology is one Blackhole P150 in a `1x1` mesh. This topology is metadata, not placement evidence.

The upstream source requires firmware 19.5.0 or newer. It reports firmware 19.6.0 as known good.

The upstream `python_env` pins `transformers==4.53.0`, `ttnn`, and `ttl==1.1.3`.

These values are source facts. They are not facts about the local machine.

## Simulator eligibility verdict

The deterministic receipt is:

`pkgs/model-target-report/receipts/qwen3.6-candidate-verdict.json`

Its status is `eligible`. Its `simulator_change_eligible` value is `true`.

This eligibility authorizes one bounded simulator change. It does not report simulator success.

The current change did not run TTsim for these reasons:

1. The Qwen tt-metal base does not match the repository package.
2. Custom `ttl` Gated DeltaNet kernels lack package-matched TTsim evidence.
3. This metadata-only change excludes the 26 checkpoint shards.

No TTsim command ran. No hardware command ran.

## Non-claims

This candidate does not establish Qwen3.6 support.

It does not establish simulator success, silicon success, physical placement, performance, profiling, thermals, or device health.

It does not establish model correctness or generated-text quality.

## Bounded next change

A later change can package the exact Knight-Ops tt-metal base and its pinned environment.

That change must start with `libttsim_bh.so` and a Blackhole `1x1` mesh.

The first smoke must cover one recurrent state carry and one full-attention cache update.

If that path is not package-matched, the later change must stop with an exact blocker.

## Validation

Run the focused checks:

```sh
nix build \
  .#checks.x86_64-linux.model-target-profile-factory \
  .#checks.x86_64-linux.model-target-candidate-report \
  --no-link
```

The checks reject authority drift, missing hybrid state, runnable Qwen profiles, and unreviewed candidate facts.
