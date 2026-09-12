# g6: the B=8 verdict straddles its bounds, and the binding is reverted

Recorded 2026-09-12 for change `qwen38-grouped-prefill-b8-rescreen`.

## Three matched runs, three adjacent identities

Each run used the frozen protocol: four roots in `single,batch,batch,single`
order, eight prompts, one warmup plus three measured runs, 64 generated tokens,
devices 0 and 1, with the grouped route gated off (the served default).

| Run | Execution identity | speedup >= 6.0 | retention >= 0.75 | TTFT <= 9.0 | tail <= 1.5 | verdict |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| baseline screen | `fypi0bf4` | 6.0642 | 0.7580 | 6.2212 | 1.2844 | admitted |
| refresh | `dswrs8nv` | 6.0492 | 0.7562 | 7.0230 | 1.2986 | admitted |
| final refresh | `nf2pd9d6` | **5.8980** | **0.7373** | 8.2544 | 1.3600 | **rejected** |

Correctness passed in all three: active prefix, clean reset, distinct state
slices, exact token parity, exact text parity.

Batch aggregate throughput is stable across the runs (100.06 / 100.16 / 100.78
tokens/second at width 8), while the single-stream root swings (16.47 / 16.31 /
17.09 tokens/second equivalent), which is what moves the two ratio gates across
their bounds. The candidate is therefore **not reliably above its predeclared
bounds**: 6.1063, 6.0642 and 5.8980 straddle 6.0, and 0.7633, 0.7580 and 0.7373
straddle 0.75.

## What that means for the admission

The 2026-09-11 admission was inside the measurement spread of the bound, not
clear of it. The honest state on the identity this change leaves behind is a
rejection, and the artifacts now say so:

- tracked receipt:
  `pkgs/qwen36/evidence/2026-09-12/qwen38-27b-batch-admission-b8-receipt.json`,
  BLAKE3 `376ee5972f989be546a80a173e0975709e1d5b70fc91a77dc8c14d8dd11799ac`,
  `admission.admitted = false` with reasons
  `aggregate-decode-speedup-below-bound` and
  `per-user-decode-retention-below-bound`;
- the earlier passing measurement is retained as history at
  `…-b8-receipt-passing-dswrs8nv.json`;
- the binding is back to `physically_admitted = false` with the current
  receipt's digest;
- the install phase now validates **consistency** instead of a fixed outcome:
  the exported `physically_admitted` must equal the tracked receipt's own
  `admitted` verdict, and the exported digest must equal the receipt's `b3sum`.
  A rejection receipt can no longer sit behind an admitted binding, and a pass
  cannot be exported while the tracked receipt records a failure.

## Non-claims

- The B=4 admission is untouched and remains the only admitted batch width.
- No batched serving route is enabled in the service; the grouped route stays
  gated off and its remaining defect is recorded in `g4-g5-blocker.md`.
- This is not a claim that B=8 is unachievable. It is a claim that the measured
  margin is inside the run-to-run spread, so admitting it would need either a
  tighter measurement discipline (more repetitions per root) or a real
  improvement that moves the ratios clear of the bounds.
