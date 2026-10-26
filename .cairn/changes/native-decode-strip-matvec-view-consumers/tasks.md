# Work record

## Performed

- Created this independent change record before implementation.
- Read root/native rules, runtime-port constraints and historical bank-matching rejection.
- [x] [serial] N1 Implement independent default-off decode strip and state-view consumers; verify independent simulator/negative controls, direct-copy removal, physical card-0 parity and checkpoint/state-carry gates documented in the evidence below. r[tenstorrent.native.decode_strip_state_views]
- [x] [serial] N2 Compare warmed physical component medians and retain both defaults off because near-flat strip results do not establish performance promotion or serving admission. r[tenstorrent.native.decode_strip_state_views]

## Qualification contract

Qualify on simulator then isolated card 0, with real 768-wide projections and 65536-way LM head, model/stateful continuation gates, independent numerical oracles and warmed median timings. Throughput/overlap is the hypothesis; fewer bytes is not required. Keep baseline defaults until numerical and performance qualification.

## Native state-view consumer evidence

- Removed the real recurrent-state materialization at the next batched WKV7
  consumer, behind `TT_WKV7_STATE_VIEW=1`; existing `Strided` metadata and parent
  lifetime are reused. Compact-state defaults are unchanged.
- Watched the focused simulator gate fail before implementation with
  `Shape("wkv7 batched state: [768x96], expected [768x64]")`.
- Scoped `step191_wkv7_state_view` simulator run with opt-in passed both gates:
  bit-identical direct/compact outputs, excluded payload-NaN poisoning,
  live signed-zero/subnormal/NaN/infinity parity, immutable parent bits,
  two recurrent tokens without state copies or downloads, and a transposed
  unsupported state view correctly materialized rather than aliased.
- Direct consumer timed trials submitted zero standalone transfer packets;
  compact consumer submitted three (two warmups and one simulated repetition).
  Both submitted 36 compute batches. Simulated timings are not silicon evidence.
- Ran the bounded state-view and matvec gates on isolated physical card 0,
  release run `1792780405`: four passed, none failed. Direct/compact state
  medians 807/817 μs; zero direct transfer packets and zero supported routed
  state copies/downloads. Unsupported transposed fallback copied exactly once.
- Matvec real checkpoint dimensions 768→768, 768→3072, 3072→768 and 768→65536
  passed independently checked outputs, but near-flat warmed medians did not
  qualify a performance promotion. Default remains off.
- Watched a wrong-row MVMUL mutation fail the simulator oracle, restored the
  production row, and reran the passing simulator and physical gates.
- Built release model binaries and passed the fixed 12-layer checkpoint oracle
  gate with each opt-in separately and both together on card 0; both flags also
  passed the accurate route and existing three-token stateful continuation.
- Exact commands, hashes, environments and raw outputs are in
  `evidence/kernel-card0-2026-10-23.json`, `evidence/model-card0-2026-10-23.json`
  and `evidence/decode-matvec.md`. Both defaults remain off; no serving admission
  or full-model performance promotion is claimed.

## Final package and integration acceptance, 2026-10-23

The full normal native package check subsequently passed: 1,556 tests passed,
zero failed, 46 existing ignored and zero filtered, with optimized host oracles
and debug assertions retained. See `native_package_resolution` in the
[installed integration evidence](../isolated-tt-lang-sim-official-skills/evidence/installed-tooling-2026-10-23.json).
The [original exact aggregate receipt](../isolated-tt-lang-sim-official-skills/evidence/final-aggregate-2026-10-23.json)
records the later quality acceptance separately from this documentation closure.
Neither package nor integration acceptance promotes the near-flat candidates
or changes the independent default-off flags.

