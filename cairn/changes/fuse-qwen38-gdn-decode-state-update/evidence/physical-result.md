# Physical result

Status: pending

## Isolated custom-operation attempt 1

The 2026-08-27 attempt used package `/nix/store/mypmph86ysm176vvjr6shqidlz1j2msd-qwen36-0.1.0` on physical Blackhole devices `0` and `1`.

The invalid 96-wide key input was rejected with the required `key_dim must be 128` diagnostic. The valid case then failed during TRISC JIT compilation. The compute source lacked explicit TRISC-aware API headers, so `CircularBuffer` and compute operations were not declared.

No numerical or performance result exists from this attempt. Production was restored and `/healthz` returned ready. The explicit-header and startup repair was rebuilt for attempt 2.

## Isolated custom-operation attempt 2

The second 2026-08-27 attempt used package `/nix/store/2vgfnik9hfvjsp43w3iqlwigrx7p54qq-qwen36-0.1.0` and runtime `/nix/store/giby29acazysmchwnwsa5ilszp9vz1dx-tt-metal-0.74.0` on the same devices.

The invalid-shape check passed again. The valid case failed during TRISC JIT compilation for two source-packaging reasons. Three init names were not part of the package-matched API: `mul_bcast_scalar_init`, `sub_init`, and `add_init`. The new-file patch hunk also declared 207 lines for a 208-line source, so the installed file lost its closing brace.

The corrected source uses `mul_tiles_bcast_scalar_init_short`, `sub_tiles_init`, and `add_tiles_init`. It starts with the reverse source mapping required by `matmul_init`, and the patch hunk now includes all 208 lines. The final package `/nix/store/acw86r5sd0yi20p55saa1sy1kik7s2h4-qwen36-0.1.0` passed all device-free checks and its new installed-source guard. Exact installed-source compilation passed for unpack, math, and pack TRISCs. A physical retry remains required.

Production was restored after attempt 2, and `/healthz` returned ready. No numerical or performance result exists from either failed attempt.

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
