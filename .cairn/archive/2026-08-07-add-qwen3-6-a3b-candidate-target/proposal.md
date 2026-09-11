## Why

The repository has one runnable RWKV-7 model target and one synthetic transformer fixture. No second real model family has exercised the candidate path.

Knight-Ops maintains a tt-nn bring-up of `Qwen/Qwen3.6-35B-A3B` for one Blackhole P150. It gates each module on PCC, measures decode and prefill rates, and records firmware and environment constraints.

This bring-up is now a reviewed upstream authority for a Qwen3.6 candidate. The repository must pin that authority and judge simulator eligibility. It must not expose a runnable target or make a support claim.

## What Changes

- Add a pinned authority manifest for the Knight-Ops bring-up tree and the Hugging Face model release.
- Extend the typed Nickel profile contract with a hybrid KV-cache and recurrent state kind, MoE layer-mix facts, and a multi-shard checkpoint layout.
- Add one non-runnable `qwen3-6-35b-a3b-p150x1-candidate` profile with hashed metadata artifacts only.
- Add device-free adapter facts for the Qwen3.5-MoE hybrid family: a required-operator inventory, a single-P150 topology, and firmware and memory machine facts.
- Extend the candidate report fixtures with reviewed Qwen3.6 facts and produce one deterministic verdict receipt.
- Add positive and negative fixtures for the contract, profile, adapter facts, and report.

## Impact

- **Packages**: `model-target-core` and `model-target-report` fixtures and candidate facts. Evidence validation stays unchanged.
- **Configuration**: typed Nickel contract and generated profile data.
- **Model authority**: exact upstream source, revision, license, metadata artifact, and reference tree identities.
- **Validation**: device-free contract, profile, report, and fixture checks.

## Out of Scope

- Fetch or hashing of the 72 GB checkpoint shards.
- ttsim execution, hardware attempts, or performance measurement.
- TT-NN operator implementation, serving, batching, or weight quantization.
- Any claim that Qwen3.6 is available, supported, or runnable.
