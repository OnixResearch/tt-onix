# Work record

## Performed

- Created this independent change record before implementation.
- Read root/native rules, runtime-port constraints and historical bank-matching rejection.
- Inspected exact TT-Metal 0.77 source and original packaged BF4 1D gate/up path; kept source simulator-only T16 restrictions separate from local logical B1/B8.
- Implemented real default-off generic-op projections and rounded Dst epilogue, actual MLP mode dispatch, explicit unsupported-decode errors, installed qualification command, and execution identity inputs.
- Preserved actual prefill dispatch, all original down-projection/collective behavior, BF4 gate/up weights, LoFi/FP32/approximate math, and independent FUSED_MLP_GATEUP_ADMITTED=False.
- Exercised installed host semantic checks: native-pack tie behavior, gate/up boundary order, B1/B8 selection, unsupported decode shapes, malformed flag, and unchanged prefill selection.
- Original attempt, 2026-10-23: executed corrected installed kernels with Blackhole TTsim 1.10.6; JIT succeeded, numerical execution stopped at `tensix_pacr: Disable_pack_zero_flags`. No workaround or simulator-only kernel branch; the later source-model resolution is recorded below.
- Parent executed isolated physical B1/B8 full TP2-local BF4 projection comparisons, then repeated the installed command. Three repeats per shape exactly matched the original BF16 product (zero unequal values, maximum error and relative L2).
- Diagnosed initial RNE-vs-native-pack discrepancy and corrected only the preserved BF16 boundary: gate/up casts use native-pack ties-away bias 0x8000/no retained-LSB term; product remains native BF16 RNE. No tolerance relaxation.
- Recorded source hashes, exact packages, source/math rationale, the original simulator blocker, physical component evidence, and slower synchronized host-wall times in dedicated docs/evidence. Candidate remains default off, with no performance/serving promotion.
- Exercised actual TP2 B8 prefill for eight distinct prompts and two traced
  recurrent continuation steps on physical IDs 0 and 1; preserved token outputs
  and source-checker identity in the dedicated B8 model receipt.
- Rebuilt the normal Qwen package with all install checks enabled; corrected
  qualification-only server imports and Python build-source namespace shadowing.
- Fresh final installed collector runs bind exact weight-index/all66-shard
  contents before/after collection. Baseline/candidate full-vocabulary original-six
  B1 mean NLL both 0.7451044766709086, all passage deltas zero, unchanged +0.010
  diagnostic passes. Earlier config-only outputs were not relabeled.
- [x] [serial] Q1 Implement the real default-off B1/B8 decode register epilogue with native BF16 boundary rounding and unchanged prefills; exercise installed semantic, exact simulator and physical component, content-bound B1 NLL and bounded B8 continuation gates recorded here. Preserve the original blocked simulator attempt separately from its later resolution. r[tenstorrent.qwen38.rounded_register_swiglu]
- [x] [serial] Q2 Build the normal Qwen package/check with all install checks enabled and retain no performance/serving promotion because installed component timings regress. r[tenstorrent.qwen38.rounded_register_swiglu]
- [x] [serial] Q3 Resolve the 2026-10-23 PACR simulator blocker in pinned TTsim 1.10.6's bounded dense-L1 model, then execute the unchanged installed B1/B8 component comparison with zero unequal BF16 values; retain native TTsim 1.10.9 and the separate GDN limitation unchanged. See [simulator receipt](../../../pkgs/qwen36/evidence/2026-10-23/rounded-register-swiglu.json), `simulator` and `original_simulator_attempt`. r[tenstorrent.qwen38.rounded_register_swiglu]

## Qualification contract

Use device-free semantic checks, simulator-before-hardware, bounded isolated silicon smoke and exact-shape timing. Bind exact package/checkpoint/command and recovered retention interface; preserve NLL +0.010 bound, old BF4 rejection, and FUSED_MLP_GATEUP_ADMITTED=False. No serving admission from component-only evidence.

## Current admission boundary

Component numerical qualification passes for B1/B8. Installed synchronized
host-wall measurements are slower than baseline (B1 0.411392→0.928172 ms;
B8 0.537680→2.236750 ms); they are not device-profiler measurements.
Full-model B1 original-six retention and bounded actual TP2 B8 state-carry gates
now pass; portable receipts are under `pkgs/qwen36/evidence/2026-10-23/`.
Normal `.#qwen38` package acceptance also passes with all install checks enabled.
No broad B8 retention, device-profiler speedup, serving admission or default
change is authorized; slower installed component host-wall timings reject
performance promotion.

## Final acceptance resolution, 2026-10-23

The source-built TTsim 1.10.6 `blackhole-pacr-zero-flags-v1` model passes
the same installed single-chip TP2-local B1/B8 numerical command. This resolves
the original PACR gate, not full-model or multi-chip simulation; no candidate
kernel branch was added. The separate GDN `ALU_ACC_CTRL` blocker remains.
The [original exact aggregate acceptance](../isolated-tt-lang-sim-official-skills/evidence/final-aggregate-2026-10-23.json)
also passed before this delivery-record reconciliation. Neither result changes
the default-off or no-performance/serving-promotion boundary above.

