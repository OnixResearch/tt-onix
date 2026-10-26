## ADDED Requirements

### Requirement: Native decode and state-view candidates preserve numerical contracts

r[tenstorrent.native.decode_strip_state_views]

The native decode matvec and WKV7 state-view routes MUST remain independent default-off opt-ins. The decode strip MUST preserve live-row arithmetic and existing padding/storage contracts; direct state consumption MUST retain the parent allocation and MUST materialize unsupported layouts through the established native path. Numerical qualification MUST distinguish simulator and physical evidence, and performance promotion MUST require matched physical measurements rather than fewer instructions or copies alone.

#### Scenario: A logical one-row projection selects strip math

- GIVEN TF32-from-F32 or BF16-from-F32 projection inputs and normal or transposed tiled weights
- WHEN the decode opt-in runs a supported M=1 projection
- THEN live outputs match the independent oracle and fractional baseline comparison, while wrong-row selection fails the numerical control.

#### Scenario: A recurrent state view reaches the real consumer

- GIVEN an eligible origin-zero F32 view into the preceding combined WKV7 result
- WHEN the state-view opt-in carries two recurrent tokens
- THEN outputs match compact consumption without standalone state transfers or downloads, parent bytes remain unchanged, and an unsupported transposed state materializes rather than aliases.

#### Scenario: Physical timings do not establish a speedup

- GIVEN passing isolated card-0 kernel, checkpoint-oracle and continuation gates
- WHEN strip projection medians are near-flat or slower than baseline
- THEN numerical qualification is retained but the default remains off and no performance or serving promotion is claimed.
