# Tasks: Return Qwen3.8-27B TP2 tuning as unselected

## Phase 1: Sweep decision

- [x] [serial] A1 Record the current TP2 configuration and fallback behavior. r[tenstorrent.qwen38_p150x2.tp2_tuning_non_admission]
- [x] [serial] A2 Record that no complete attributed profile or measured candidate sweep exists. r[tenstorrent.qwen38_p150x2.tp2_tuning_non_admission]
- [x] [serial] A3 Return this TP2 tuning sweep with no selected winner or speedup claim. r[tenstorrent.qwen38_p150x2.tp2_tuning_non_admission]

## Phase 2: Configuration boundary

- [x] [serial] S1 Keep all current program and collective configurations unchanged. r[tenstorrent.qwen38_p150x2.tp2_tuning_non_admission]
- [x] [serial] S2 Add no generated TP2 winner data and do not call current fallbacks tuned. r[tenstorrent.qwen38_p150x2.tp2_tuning_non_admission]
- [x] [serial] S3 Require a new validated Cairn change before future TP2 tuning admission. r[tenstorrent.qwen38_p150x2.tp2_tuning_non_admission]

## Phase 3: Evidence and verification

- [x] [parallel] V1 Add the positive incomplete-sweep no-winner scenario. r[tenstorrent.qwen38_p150x2.tp2_tuning_non_admission.valid]
- [x] [parallel] V2 Add the negative scenario that isolated data does not imply model admission. r[tenstorrent.qwen38_p150x2.tp2_tuning_non_admission.invalid]
- [x] [serial] D1 Record current behavior, missing evidence, and all non-claims. r[tenstorrent.qwen38_p150x2.tp2_tuning_non_admission]
- [x] [serial] G1 Run Cairn validation, lifecycle gates, sync checks, and the drain evidence guard. r[tenstorrent.qwen38_p150x2.tp2_tuning_non_admission]

## Verification Coverage

- `Scenario: TP2 sweep evidence is incomplete` -> A1, A2, A3, V1
- `Scenario: An isolated operation result becomes available` -> S3, V2
