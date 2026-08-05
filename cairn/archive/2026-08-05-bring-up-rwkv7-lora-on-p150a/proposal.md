## Why

The CPU reference can define correct RWKV-7 LoRA behavior, but it cannot prove TT-XLA compilation or P150A execution.

RWKV-7 has a recurrent WKV7 sequence equation and a backward path that TT-Blacksmith does not currently list. A full launch can hide the first unsupported operation without focused compiler probes.

This change establishes the smallest complete accelerator claim: one admitted SFT batch, full loss, adapter gradients, and one optimizer step on one logical P150A device.

## What Changes

- Add focused TT-XLA compile probes for the naive WKV7 recurrence, its backward path, LoRA projections, loss, clipping, and optimizer update.
- Add a fixed-shape RWKV-7 LoRA training graph for the pinned World2.8 0.1B checkpoint.
- Add one logical `1x1` P150A execution profile that runs inside the validated dual-card platform.
- Add placement evidence that rejects unsupported host fallback in the model and optimizer graph.
- Add CPU-to-P150A comparisons for logits, loss, selected activations, adapter gradients, and adapter deltas.
- Add bounded one-step, multi-step, save, load, and resume hardware runs.
- Add positive and negative device-free fixtures for compile, placement, comparison, and hardware receipt admission.

## Impact

- **Packages**: a TT-XLA RWKV-7 LoRA experiment package and focused compiler probes.
- **Runtime**: one logical P150A device from the admitted training platform.
- **Model**: static-shape sequence training for the pinned RWKV-7 0.1B checkpoint.
- **Evidence**: graph, placement, numerical comparison, optimizer, artifact, health, and restoration receipts.
- **Validation**: device-free receipt fixtures plus bounded P150A execution.

## Out of Scope

- Two-card data parallelism or model parallelism.
- Full-parameter fine-tuning, QLoRA, long-context optimization, or custom native TT-Metal kernels.
- General RWKV-7 model-size support.
- Throughput targets or claims that P150A is faster than the CPU reference.
