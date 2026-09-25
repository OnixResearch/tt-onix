# Masked prefill traces on P150x2

Recorded 2026-09-25 on devices 0 and 1, runtime `tt-metal-0.77.0`, with the
fused GDN decode step. The throwaway harnesses are in `.pi/tmp/gdn-step/`
(`prefill_trace_eval2.py`, `trace_iso*.py`, `trace_v*.py`, and
`service_traced.sh`). The host was shared, with a load average of about 27 to 95.

## Device time versus eager

A captured 128-token prefill replays in 0.199 to 0.200 s. Eager prefill of the
same prompt took 2.1 to 3.7 s, almost all of it host dispatch.

## Exactness

- **Model level.** Traced and eager first-token logits and GDN recurrent state
  matched with a max abs diff of 0.0 in every comparison run:
  - prompts of 8, 11, and 200 tokens (both buckets);
  - an exact 256-token full bucket, compared in a fresh process;
  - the 11-token prompt after an eager run and after a decode trace was used
    and released.
  A later run in one process was cut short by the recapture hang below, after 8
  and 200 tokens had matched.
- **Adapter sequence.** A 14-batch sequence ran in one process and was compared
  with an eager reference from another process. Widths were 1, 4, 5, 8, 11,
  and 16, and the batches mixed 200-, 300-, 900-, and 1,450-token prompts. The
  first token plus 6 decoded tokens matched for every user in all 14 batches.
- **Order bug.** Before the fix, bucket 256's inputs were allocated after the
  bucket-128 capture. A batch that replayed 128 before 256 then diverged,
  because its RoPE tables sat in the 128 trace's scratch. After allocating every
  input first, all 14 batches matched.
- **Page-table width.** An eager reference with request-bounded page-table rows
  and one with full-width rows gave identical tokens in all 14 batches.

## Ordering hazards found on silicon

| Order | Result |
| --- | --- |
| capture, compile decode width 1, release, replay, capture decode | device hang in the decode capture |
| warm width 1, capture, first-time width 8, replay, width 8 again | hang |
| capture, replay, compile decode, recapture | hang while capturing bucket 256 |
| warm all widths, eager shapes, and slot installs, then capture once | exact, no hang, no new allocator blocks, program count constant |

After the first-time width-8 decode, the allocator showed 124 new DRAM blocks.
Most were kernel binaries at the top of DRAM, but six sat in the region where
the trace scratch lives. Hence the rule: capture after every reachable
program, and release the traces on any later compile.

## Packaged service

Package `/nix/store/09r8s1g67jdpswqr8137xygiy81fq8i7-qwen38-0.1.0`. Each
setting ran with default serving (step decode, grouped prefill), traced prefill
off and then on. Each width ran three rounds of concurrent 64-token greedy
requests.

| Width | Worst TTFT, off | Worst TTFT, on | Aggregate tok/s, off | Aggregate tok/s, on |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 2.34–2.58 s | 0.46–0.59 s | 9.5–10.2 | 14.3–15.7 |
| 4 | 3.00–3.83 s | 2.08–2.40 s | 32.9–38.3 | 40.8–43.8 |
| 8 | 6.19–7.70 s | 4.44–4.69 s | 25.9–50.8 | 59.0–60.7 |
| 11 | 10.4–32.3 s | 5.84–6.56 s | 15.3–45.1 | 60.0–67.3 |
| 16 | 12.9–13.7 s | 8.60–9.37 s | 38.0–56.2 | 69.4–73.5 |

- Generated text was identical with the setting off and on in 45 of 45 batches
  (every width and round).
- A 330-token prompt then a short one: the short prompt stayed traced
  (0.50 s), and no `masked-prefill-traces-released` event occurred.
- Startup: the width-1 and width-4 warm-up took 12.8 s, and the traces were
  captured 221.7 s later.
- Width 11 now serves. Before this change it failed decode trace capture on the
  KV-update shard grid.

## In-trace install and buckets up to 1,024

Recorded 2026-09-25 (host load average about 27–30).

### Where the per-user time went

At width 8, the earlier traced path spent its time as follows:

- about 0.21 s per user waiting for the replay's logits;
- 0.30 to 0.35 s per user in host `write_slot` dispatch;
- 0.035 s per user in 240 `ttnn.clone` calls.

The install cost more than the replay.

### The install program

`slot_install` reads the slot from a staged device buffer. It copies:

- recurrent-state tile slabs to `slot * slab_tiles`;
- conv-state and logit rows into row `slot` of a tile row.

Device check (`.pi/tmp/gdn-step/test_slot_install.py`):

- a 24×128×128 fp32 slab, four 1×5120 bf16 conv rows, and a 248,320-wide
  logits row matched torch bit for bit;
- slots 0, 5, and 15 were checked eagerly, and slots 3, 12, and 9 through one
  captured trace with the slot restaged;
- one replay of that install took 0.31 ms.

### Buckets 512 and 1,024

Traced and eager logits matched exactly (max abs diff 0.0) at 300, 512, 700,
900, and 1,024 tokens. Prefill time:

| Prompt tokens | Eager | Traced |
| ---: | ---: | ---: |
| 300 | 2.65 s | 0.38 s |
| 512 | 2.30 s | 0.39 s |
| 700 | 1.91 s | 0.58 s |
| 900 | 2.00 s | 0.57 s |
| 1,024 | 2.33 s | 0.57 s |

With the four prefill traces resident and no decode trace, the trace region
used 70.5 of 134.2 MB per bank.

### Adapter sequence

The same 14-batch sequence ran traced and was compared with an eager reference
from a separate process. All 14 batches were exact. The only batch still
prefilled eagerly was the 1,450-token one; all others were traced.

| Batch | Eager TTFT | Traced TTFT |
| --- | ---: | ---: |
| w1 | 4.32 s | 0.23 s |
| w4 | 14.2 s | 0.85 s |
| w8 | 41.5 s | 1.69 s |
| w11 | 29.5 s | 2.33 s |
| w16 | 75.9 s | 3.47 s |
| w2 at 900 tokens | 4.02 s | 0.78 s |

### Packaged service

Package `/nix/store/jgc3zkkk72q1hp1yaam7bcayywh8pv50-qwen38-0.1.0`, with traced
prefill off and then on, three rounds per width:

| Width | Worst TTFT, off | Worst TTFT, on | Aggregate tok/s, off | Aggregate tok/s, on |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 2.20–2.61 s | 0.22 s | 9.6–10.4 | 16.4–17.1 |
| 4 | 3.57–3.78 s | 0.85 s | 33.4–34.6 | 54.9–57.1 |
| 8 | 6.31–7.18 s | 1.69 s | 27.9–50.2 | 91.8–94.3 |
| 11 | 8.88–33.4 s | 2.32 s | 14.6–49.7 | 94.7–108.7 |
| 16 | 10.8–17.6 s | 3.36–3.37 s | 32.5–63.7 | 115.0–119.5 |

- A 330-token prompt took 15.9 s with traced prefill off and 0.34 s with it on.
- Generated text was identical with the setting off and on in 120 of 120
  responses.
- The four traces were captured 214.5 s into startup.

## Mixed batches and prompts above 1,024 tokens

Recorded 2026-09-25.

### Mixed batches

A batch that mixes traced and eager users now runs the eager users first,
including their state installs. It then checks for a first-time compile
(releasing the traces if one happened) and replays the rest. Before, one long
prompt made the whole batch eager. With the sequence below, the program-cache
entry count stayed constant.

### Long prompts

Long prompts replay the 1,024 bucket from a sequence start, then the smallest
tail bucket as a continuation. Chunk start, RoPE rows, page-table slice, and a
keep-state predicate are staged per replay. The trace starts with
`ttnn.where(keep, state, zero, output_tensor=state)` on each GDN recurrent state
and conv carry. A device test confirmed that this in-place broadcast select is
bit-exact both ways: keep 0 yields +0.0, and keep 1 preserves -0.0 and every
other bit. So one set of four traces serves every prompt up to 2,047 tokens.

### Adapter sequence

The sequence had 20 batches and ran in one process. It was compared with an
eager reference from another process, and all 20 batches were exact. The
long-prompt and mixed batches:

| Batch | Eager TTFT | Traced TTFT |
| --- | ---: | ---: |
| w1, 1,450 tokens | 8.2 s | 0.92 s |
| w1, 2,000 tokens | 10.7 s | 1.12 s |
| w4 with one 1,450-token prompt | 13.0 s | 1.53 s |
| w8 with one 1,450-token prompt | 121.7 s | 2.37 s |
| w3: 1,100, short, 2,000 | 20.0 s | 2.08 s |

### Packaged service

Package `/nix/store/pqasgm6hj8k32vhmss0xd0a1pm2gv34v-qwen38-0.1.0` ran with
traced prefill off and then on. The host load average was about 32.

| Batch | Worst TTFT, off | Worst TTFT, on |
| --- | ---: | ---: |
| w1, about 1,480 tokens | 6.3–66.1 s | 0.80–0.82 s |
| w8 with one about-1,480-token prompt | 29.0–32.7 s | 2.27–2.28 s |
| w1 short | 5.1–5.7 s | 0.25–0.28 s |
| w16 short | 20.0–27.6 s | 3.37 s |

- Generated text was identical with the setting off and on in 121 of 121
  responses.
- No trace release occurred.
- Capture took 406.6 s of startup under this load.
