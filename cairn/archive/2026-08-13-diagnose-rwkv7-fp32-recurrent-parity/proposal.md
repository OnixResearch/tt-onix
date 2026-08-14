## Why

The HiFi4 BF16 model-matmul control restores all sixteen pinned CPU tokens. Its final FP32 recurrent-matrix identity still differs from the CPU authority.

Window `16` must remain closed until a bounded diagnosis finds the first recurrent-matrix divergence and proves its numerical mechanism.

## What Changes

- Add aggregate-only CPU and device recurrent-matrix boundary identities.
- Find the first differing model step, layer, and recurrence operation.
- Test distinct numerical mechanisms with the smallest bounded discriminator.
- Keep candidate v3 absent and production windows `[2, 4, 8]` unless exact parity passes.

## Impact

- **Files**: RWKV-7 CPU authority, diagnostic runtime, evidence validation, Nix package wiring, Cairn evidence, and the accepted runtime specification
- **Testing**: positive and negative unit tests, package checks, source and closure checks, bounded device-free controls, and no more than three bounded physical discriminators
