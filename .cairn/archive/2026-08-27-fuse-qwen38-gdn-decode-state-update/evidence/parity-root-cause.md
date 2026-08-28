# GDN parity root-cause search

## Success contract

Goal: keep the `full` mode's measured speed while reproducing the control token sequence and recurrent state behavior.

Completion requires all of these results:

- The fused recurrence matches the package-matched composite recurrence on physical hardware.
- The full-model screen keeps the control token sequence.
- Exact token and text parity, state isolation, and clean reset pass.
- The fixed GDN ratio `<= 0.95` and full-step ratio `<= 0.97` still pass.

False completion includes Torch-only tolerance, one-step PCC, speed without token parity, stale trace reuse, or evidence from another package identity.

Risks include FP32 operation order, matmul accumulation mode, pack/unpack boundaries, TP2 tensor mapping, in-place state aliasing, and trace state progression.

Budget: package-matched sources only and no pin change. The initial three-run budget was extended for three discriminating follow-ups after new physical evidence identified decay precision, mixed-kernel unpack mode, and layer-subset mechanisms. The allowed outcomes are validated, blocked, or exhausted.

The passes below are serial and correlated because subagent consent was not granted.

## Approach registry

| Family | Mechanism | Claim | Artifact | Next check | State |
|---|---|---|---|---|---|
| Arithmetic equivalence | Compare the fused kernel with the actual TTNN composite from identical inputs and independent state buffers. | Distinguishes semantic or rounding drift from model integration. | One-chip A/B receipts and generated unpack descriptors. | Requires a staged kernel architecture that preserves each TTNN precision boundary. | blocked |
| TP2 mapping | Exercise the repaired conversion through the model's two-device tensor distribution. | Finds local-head indexing or multi-device alias defects hidden by the one-chip smoke. | P150x2 full-model screen and balanced roots. | None in this bounded change. | validated |
| Trace lifecycle | Compare eager state progression with captured replay from the same initialized state. | Finds trace-time alias or update-order drift. | Distinct slices and clean reset passed in four B=4 roots. | None in this bounded change. | validated |
| Identity contamination | Separate processes and receipts bind the same package, runtime, checkpoint, tokenizer, mesh, and CPUs. | A stale execution identity did not cause the token change. | Four-mode screen receipts. | None. | falsified |
| Decay arithmetic | Compare full and decay-only recurrence, then match the binary-SFPU route. | Tests whether decay alone explains long-run drift. | Decay error fell, but full-state and long-output divergence remained. | No live repair remains in this kernel shape. | exhausted |
| Mixed unpack precision | Compare TTNN and fused JIT descriptors and test identity decay. | Finds precision loss from FP32-to-TF32 unpack inside the mixed matmul kernel. | Identity-decay error and per-CB descriptor evidence. | Requires aliases or staged kernels; direct mode in the mixed kernel corrupted state. | blocked |
| Layer subset | Fuse alternating GDN layers to bound accumulated drift. | Tests whether a partial fusion can preserve exact output and the speed gates. | Physical 24-layer long-output screen. | No weaker subset can establish semantic exactness. | falsified |
| Margin sensitivity | Small recurrence drift accumulates through GDN layers and crosses later argmax margins. | Explains long-output divergence despite high isolated PCC. | Divergence at generated indices `47`, `32`, and `16` for three prompts. | None. | validated |

## Initial evidence

The physical screen kept token `13` for `off`, `norm`, and `state`. The `full` mode produced token `1088`. Its GDN ratio was `0.8739084261`, and its full-step ratio was `0.9412087916`. Speed passed, but parity failed.

## Discriminating result 1: physical tile layout

The custom reader assigns one full tile sequence to each head. The model supplies Q, K, V, beta, and decay as `[B,1,H,D]`, where `H` is packed into tile rows. The original route used metadata-only reshapes to `[B*H,1,D]`. The reader then treated packed storage as one full tile sequence per head and read beyond the valid input pages.

A one-chip A/B test reproduced the defect. The fused state reached magnitudes up to `7.908906574920249e+37`, while the composite state remained below `0.036`. This is a semantic layout defect, not ordinary FP32 drift.

Materializing `[B,H,1,D]` with `ttnn.permute` before flattening removed the corruption. After two steps, state PCC was `0.9999659061431885`, output PCC was `0.9999662041664124`, and the maximum state error was `0.0004925858229398727`. The output requires the inverse permutation for the same reason.

The package patch now includes explicit head-major input conversion and packed-head output restoration. Exact element equality is not claimed by the one-chip A/B result.

## Discriminating result 2: P150x2 model integration

The first rebuilt route reached trace capture but rejected beta and decay because the model supplies rank-three `[B,1,H]` tensors. The final route lifts both tensors to rank four before the same materialized permutation.

Package `/nix/store/dkwhljanp15jb35jsfx12kbzc2l0j1r4-qwen36-0.1.0` passed all device-free checks. Its P150x2 screen retained control tokens `11751` and `13`. The GDN ratio was `0.9106139348381834`, and the full-step ratio was `0.9384588001106107`. Both fixed speed gates passed.

The root cause is validated for the bounded screen. Balanced duplicate roots subsequently passed, but long-output parity remained open.

## Discriminating result 3: long decode and decay arithmetic

A mixed `single-off,batch-full,batch-full,single-off` B=4 run passed all performance, state-slice, active-prefix, and cleanup gates. It reached `62.98632743556621` aggregate tokens per second, `3.734619848908698` aggregate speedup, and `0.9336549622271745` per-user retention.

Exact 64-token and text parity failed. Three prompts first diverged at generated indices `47`, `32`, and `16`; one prompt stayed exact for all 64 tokens.

The decay-only physical A/B test reproduced nearly all recurrence drift. After two steps, its maximum state error was `0.0005043745040893555`, compared with `0.0004925858229398727` for the full recurrence A/B. Decay is therefore the dominant arithmetic mechanism.

The composite path fills the scalar decay tile and uses the binary-SFPU multiply path. Matching that route reduced the two-step full-state error, but long output still diverged.

## Adversarial follow-ups

An identity-decay test showed that the mixed matmul kernel unpacks FP32 inputs to TF32. The package's standalone FP32 binary kernels instead use `UnpackToDestFp32`. Enabling that mode directly inside the mixed kernel corrupted state because its direct-to-destination inputs later share one kernel with SrcA/SrcB matrix stages. A tighter alias experiment preserved identity state exactly, but corrupted every nonzero state element after matmul-to-SFPU transitions. Both experiments were removed.

Applying exact decay in a separate TTNN launch did not remove state-update drift. Fusing alternating GDN layers also failed: three of four 64-token prompts still changed output. Partial fusion cannot establish general semantic exactness.

## Terminal result

The search is **exhausted** for the declared single-kernel design. The candidate satisfies bounded speed, spread, state-isolation, and reset gates, but fails exact token and text parity. A valid follow-up needs a different staged architecture with explicit FP32 aliases or separate precision-preserving kernels. That stronger design is outside this change.

The final candidate remains default-off and is rejected for admission. No production or deployment claim is made.
