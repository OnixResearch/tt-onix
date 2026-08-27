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

Budget: package-matched sources only, no pin change, three bounded physical diagnostic runs, and one rebuild for each distinct repair. Balanced roots and admission remain blocked until parity returns. The allowed outcomes are validated, blocked, or exhausted.

The passes below are serial and correlated because subagent consent was not granted.

## Approach registry

| Family | Mechanism | Claim | Artifact | Next check | State |
|---|---|---|---|---|---|
| Arithmetic equivalence | Compare the fused kernel with the actual TTNN composite from identical inputs and independent state buffers. | Distinguishes semantic or rounding drift from model integration. | One-chip two-step A/B receipts before and after explicit layout conversion. | Rebuild and repeat the full P150x2 screen. | validated |
| TP2 mapping | Exercise the repaired conversion through the model's two-device tensor distribution. | Finds local-head indexing or multi-device alias defects hidden by the one-chip smoke. | P150x2 full-model screen. | Continue with balanced roots. | validated |
| Trace lifecycle | Compare eager state progression with captured replay from the same initialized state. | Finds trace-time alias or update-order drift. | Captured full-decode replay retained token `13` across five samples. | Extend to long-decode and reset checks. | active |
| Identity contamination | Separate processes and receipts bind the same package, runtime, checkpoint, tokenizer, mesh, and CPUs. | A stale execution identity did not cause the token change. | Four-mode screen receipts. | Recheck only if another mechanism is falsified. | falsified |
| Margin sensitivity | Small recurrence drift accumulates through 48 GDN layers and crosses the second-token argmax margin. | Explains a token change despite high isolated PCC. | Layer/state error growth and final logit margin. | Inspect only after direct A/B error size is known. | active |

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

The root cause is validated for the bounded screen. Balanced duplicate roots, exact text, long decode, isolation, reset, and admission remain open.
