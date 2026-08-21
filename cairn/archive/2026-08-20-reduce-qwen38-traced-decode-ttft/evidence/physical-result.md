# Physical result

## Verdict

The request-bounded traced-decode candidate is rejected.
Production remains eager.

The balanced run used physical PCIe device IDs `0` and `1` in this order:

`eager, candidate, candidate, eager`

Each root used four prompts, one warm-up, three measured runs, and `64` generated tokens.

## Admission result

- Exact generated token IDs: pass
- Exact generated text: pass
- Clean reset: pass
- Identity match: pass
- Candidate prefill table width: `2` blocks
- Candidate trace-bound decode table width: `32` blocks
- Median decode speedup ratio: `6.942337116061393`
- Tail-latency ratio: `0.046057147610702766`
- TTFT ratio: `1.1069467490614955`
- Maximum admitted TTFT ratio: `1.05`
- Rejection reason: `ttft-regression-above-bound`

Receipt BLAKE3:

`8e5d99ce972f739548b04cb57bf19c2199e26fefbf3fce36fa9f50406a457a59`

Receipt path:

`pkgs/qwen36/evidence/2026-08-20/qwen38-27b-traced-decode-ttft-candidate-receipt.json`

## Phase attribution

The candidate median TTFT was `1.1874256309965858` seconds.
The eager median TTFT was `1.0727034809970064` seconds.

The residual difference was concentrated in device prefill:

- Candidate median device prefill: `1.171225828002207` seconds
- Eager median device prefill: `1.0575300640048226` seconds
- Candidate median first-token readback: `0.018084625495248474` seconds
- Eager median first-token readback: `0.016442937005194835` seconds

Request reset and trace preparation remained outside the unchanged TTFT interval.
Their candidate medians were `0.008478285002638586` and `0.0013550005096476525` seconds.

The bounded table was not sufficient to meet the TTFT gate.
This result does not prove which prefill operation caused the remaining device-time difference.

## Disposition

The package keeps the candidate explicit and unadmitted.
A prefill-width change releases the parked trace before prefill, which prevents post-park compilation from clobbering that trace.
The next trace is captured after the unchanged TTFT interval.

A later change must add operation-attributed physical prefill profiling before it selects another optimization.
It must not infer a cause from this aggregate phase result.

## Non-claims

- No production traced-decode admission
- No concurrent-request throughput claim
- No prefill speedup claim
- No claim for a mesh other than physical P150x2
- No claim for another generated length
