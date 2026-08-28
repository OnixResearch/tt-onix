# Physical result

Status: rejected

## Isolated custom-operation attempt 1

The 2026-08-27 attempt used package `/nix/store/mypmph86ysm176vvjr6shqidlz1j2msd-qwen36-0.1.0` on physical Blackhole devices `0` and `1`.

The invalid 96-wide key input was rejected with the required `key_dim must be 128` diagnostic. The valid case then failed during TRISC JIT compilation. The compute source lacked explicit TRISC-aware API headers, so `CircularBuffer` and compute operations were not declared.

No numerical or performance result exists from this attempt. Production was restored and `/healthz` returned ready. The explicit-header and startup repair was rebuilt for attempt 2.

## Isolated custom-operation attempt 2

The second 2026-08-27 attempt used package `/nix/store/2vgfnik9hfvjsp43w3iqlwigrx7p54qq-qwen36-0.1.0` and runtime `/nix/store/giby29acazysmchwnwsa5ilszp9vz1dx-tt-metal-0.74.0` on the same devices.

The invalid-shape check passed again. The valid case failed during TRISC JIT compilation for two source-packaging reasons. Three init names were not part of the package-matched API: `mul_bcast_scalar_init`, `sub_init`, and `add_init`. The new-file patch hunk also declared 207 lines for a 208-line source, so the installed file lost its closing brace.

The corrected source uses `mul_tiles_bcast_scalar_init_short`, `sub_tiles_init`, and `add_tiles_init`. It starts with the reverse source mapping required by `matmul_init`, and the patch hunk now includes all 208 lines. The final package `/nix/store/acw86r5sd0yi20p55saa1sy1kik7s2h4-qwen36-0.1.0` passed all device-free checks and its new installed-source guard. Exact installed-source compilation passed for unpack, math, and pack TRISCs.

Production was restored after attempt 2, and `/healthz` returned ready. No numerical or performance result exists from either failed attempt.

## Isolated custom-operation attempt 3

The third 2026-08-27 attempt used the final package on physical Blackhole hardware. Topology discovery found devices `0` and `1`, firmware `19.11.0`, and KMD `2.10.0`. The bounded smoke opened a `1x1` mesh.

The invalid 96-wide key input was rejected with the required diagnostic. The valid 24-head, 128-by-128 FP32 recurrence then completed two state-carry steps. The minimum output PCC was `0.9999610781669617`, and the minimum state PCC was `0.9999649524688721`. Maximum absolute errors were `0.00012158090248703957` for output and `0.0004848763346672058` for state. The returned state kept the same L1 buffer address.

The isolated physical verdict is `pass`. This proves physical JIT execution and bounded two-step numerical correctness. It does not prove full-model performance, exact generated output, long-decode stability, reset behavior, or production admission.

By user direction, production remains inactive and runtime-masked while the remaining physical tests run.

## Balanced component result

The final package is `/nix/store/dkwhljanp15jb35jsfx12kbzc2l0j1r4-qwen36-0.1.0`. The physical order was `control,candidate,candidate,control` on devices `0` and `1`. All roots captured and replayed the full decode trace, cleaned up, and retained tokens `11751` and `13`.

The candidate GDN median ratio was `0.8859209237581226`, below the fixed `0.95` maximum. The candidate full-step median ratio was `0.9389105633859162`, below the fixed `0.97` maximum. This is a `1.0650641701098582` full-step speedup.

The largest duplicate-root spread was `1.0224085594579935`, below the fixed `1.25` maximum. The bounded component verdict is `pass`.

## Long parity and B=4 admission

The B=4 order was `single-off,batch-full,batch-full,single-off`. All measured responses used 64 generated tokens and the same package, runtime, checkpoint, tokenizer, mesh, and devices.

State and lifecycle checks passed. Batch state slices were distinct and formed the active prefix. Every measured request reported clean reset. Aggregate decode reached `62.98632743556621` tokens per second, `3.734619848908698` speedup, and `0.9336549622271745` per-user retention.

Exact output failed. Three prompts first diverged at generated token indices `47`, `32`, and `16`; one prompt stayed exact through all 64 tokens. The receipt rejected admission with `token-parity-failed` and `text-parity-failed`.

Follow-up decay matching reduced isolated state error but did not restore long-output parity. Direct FP32 unpack was incompatible with this mixed matmul kernel and was removed. An alternating 24-layer fusion also changed long outputs. No candidate mode satisfied both the fixed speed gates and exact-output requirements.

The final decision is **reject**. The custom recurrence remains default-off. No deployment or production admission is authorized.

By user direction, production remains inactive and runtime-masked.

The fixed order is `control,candidate,candidate,control` on physical devices `0` and `1`.
The fixed candidate modes are `norm`, `state`, and `full`.

## Required evidence

- Package, runtime, checkpoint, tokenizer, mesh, and device identities.
- Exact generated tokens and text.
- Distinct recurrent and KV state slices.
- Clean reset and cleanup.
- At least five retained GDN and full-step samples per root.
- Duplicate-root spread at most `1.25`.
- Candidate GDN median ratio at most `0.95`.
- Candidate full-step median ratio at most `0.97`.
- Fresh B=4 admission for a selected identity.

## Non-claims

- Static operation counts do not prove speedup.
- Independent TT-Lang results do not transfer to TT-Metal 0.74.
- A component win does not admit production.
- No TT-Metal pin change is included.
- The result does not admit or deploy the custom recurrence.
