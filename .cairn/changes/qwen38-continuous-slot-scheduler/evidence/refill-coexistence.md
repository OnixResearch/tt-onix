# Refill coexistence with a resident decode trace (2026-09-27)

Diagnostic evidence for the design's first refill question: can a new request be
prefilled into a free slot while the batched decode trace stays resident? This is
not a performance or admission claim, and it does not replace tasks s1–s6.

## Why it was in doubt

The adapter releases every decode trace before a prefill or a structured read.
A trace replay writes to the addresses recorded at its capture, so it can overwrite
buffers allocated after that capture.

The masked prefill traces are captured at startup, and the decode trace later,
per group. So a decode replay's intermediates avoid the prefill traces' persistent
buffers. The buffers at risk are the ones the decode trace allocated after the
masked capture:

- **Inputs.** Tokens, positions and rope are restaged every step. The page table
  is restaged only when a trace is captured or reused.
- **Outputs.** Logits and winners are read right after each replay.

Hypothesis: a refill needs only a full restage of the decode inputs afterwards.

## Method

The probe is `scripts/qwen38-refill-coexistence-probe.py`, run on britton-desktop
(P150x2).

- **Packages.** qwen38 `/nix/store/6j5h0gvpr5k97yjfb9nak7ba3vifdg6k-qwen38-0.1.0`
  and tt-metal `/nix/store/7hplkynh047dsk98cwqpfsaaq6lwyy0m-tt-metal-0.77.0`
  (tt-onix `4694f26`).
- **How it ran.** `qwen38-p150x2.service` was stopped, and the probe ran as a
  transient unit with that service's environment. Filler text:
  `.pi/references/rwkv-7/rwkv-7-goose-arxiv-2503.14456v2.txt`.
- **Warm-up.** The same as the service: widths 1 and 4, then every decode width
  and the masked prefill traces (buckets 128–1024, fusion mode `step`).
- **Prompts.** A, B, C and D are chat prompts of 23, 1,523, 3,022 and 2,521 tokens.
- **Reference.** A–D are prefilled together, then 48 greedy steps run at width 4.
- **Refill runs.**
  1. A–C are prefilled into slots 0–2, and the width-4 decode trace is captured
     with slot 3 idle.
  2. 24 steps run.
  3. D is prefilled into slot 3 with `prefill_paged_slots` while the decode trace
     stays resident.
  4. All four decode inputs are restaged from the host.
  5. 24 more steps run with all four rows.
  - R2 refills with the masked prefill traces; R1 refills eagerly.

## Result

| Run | A, B, C equal the reference (49 tokens each) | D equals its reference (25 tokens) | Trace binding | Masked traces | New programs | D prefill |
| --- | --- | --- | --- | --- | --- | --- |
| R2 masked-trace refill | yes | yes | held | intact | 0 | 1.489 s |
| R1 eager refill | yes | yes | held | intact | 0 | 11.625 s |

The first divergence was none in every row of both runs. The probe exited
cleanly. D's refilled answer began "Based on the passage provided, here are two
technical terms: 1. **Householder matrix**", the same as its reference.

## What it settles, and what it doesn't

**Settled.** The design's first refill candidate works without changing the
capture order. The masked prefill replay can install a new user's state into a
free slot while the decode trace stays resident. The condition is that the
scheduler restages every decode input afterwards, including the page table. The
masked path was 7.8 times faster than eager prefill for D.

**Not covered:**
- sampled routes;
- the fed-back greedy stream (device-side token feedback);
- refilling a slot that had held an earlier user;
- more than one refill per group;
- structured reads interleaved with a resident trace;
- the page-table restage cost at width 16;
- queue-inclusive latency or fairness.

Tasks s1–s6 still stand, and this result feeds s5.
