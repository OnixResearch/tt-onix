## Why

The repository has a pinned RWKV-7 World2.8 0.1B checkpoint and a CPU inference equation reference. It has no training graph, supervised fine-tuning data contract, LoRA plan, gradient oracle, optimizer step, or adapter checkpoint contract.

Tenstorrent bring-up needs a device-free authority first. Without this authority, a finite accelerator loss or gradient cannot establish RWKV-7 fine-tuning correctness.

## What Changes

- Add a CPU FP32-from-BF16 RWKV-7 sequence reference for the pinned World2.8 0.1B checkpoint.
- Add the paper-matched naive WKV7 forward and backward equations for bounded sequence fixtures.
- Add an explicit LoRA target policy for RWKV-7 attention and feed-forward projection matrices.
- Add typed Nickel configuration for the model, tokenizer, SFT data, sequence profile, LoRA, optimizer, seed, and checkpoint policy.
- Add deterministic JSONL preprocessing with prompt-loss masking and the pinned RWKV World tokenizer.
- Add one-step and resume fixtures that prove base-weight immutability, adapter updates, finite loss, and deterministic artifacts.
- Add a Safetensors adapter format with a BLAKE3 manifest and explicit non-claims.

## Impact

- **Packages**: a new device-free RWKV-7 LoRA reference package and focused checks.
- **Model authority**: exact checkpoint, tokenizer, source, equation, LoRA target, and precision identities.
- **Configuration**: typed Nickel source with generated runner data.
- **Data**: local JSONL SFT input with deterministic token and label generation.
- **Validation**: positive and negative forward, backward, optimizer, checkpoint, and resume fixtures.

## Out of Scope

- TT-XLA compilation or Tenstorrent device access.
- Full-parameter fine-tuning, QLoRA, state tuning, DPO, RLHF, or pre-training.
- Models other than `RWKV/RWKV7-Goose-World2.8-0.1B-HF` at its pinned revision.
- Long-context performance or production dataset quality claims.
