# Qwen3.8 fused GDN decode step

The P150x2 Qwen3.8-27B service decodes every GDN layer through two custom
programs by default. It no longer runs the composite chain of about 60 TTNN
operations. The Cairn change is
`.cairn/changes/serve-qwen38-fused-gdn-decode-step`. Its evidence directory holds
the raw measurements.

## What runs

For each GDN layer, decode runs these operations:

1. The existing `[qkv|z|a|b]` projection.
2. `conv_prep`: depthwise causal conv and SiLU for q/k/v, SiLU for the output
   gate, and an fp32 copy of the a/b logits into one persistent buffer. The
   conv state shifts in place.
3. `recurrence`: one instance per (batch, value head). It computes the q/k L2
   norms, beta, decay, the gated delta rule, the output read, and the gated
   RMSNorm. It writes the out projection's bf16 input and updates the fp32
   recurrent state in place.
4. The existing out projection and reduce-scatter.

The kernels live in `models/demos/blackhole/qwen36/tt/gdn/kernels/` in the
package tree. `pkgs/qwen36/gdn-decode-step.patch` adds them. They are
JIT-compiled through `ttnn.generic_op`, so the shared TT-Metal runtime does not
change.

## Measured effect (B=1, P150x2, 2026-09-24)

| Route | Decode trace replay | Per token incl. host | tok/s |
| --- | ---: | ---: | ---: |
| composite (`off`, previous default) | 54.3 ms | 59.0 ms | 16.9 |
| fused recurrence (`full`) | 50.9 ms | 55.3 ms | 18.1 |
| fused step (`step`, new default) | 42.7 ms | 45.6 ms | 22.0 |

The packaged service's own per-token latency, over 976 greedy tokens, moved
from 59.9 ms (16.7 tok/s) to 46.6 ms (21.5 tok/s).

Greedy tokens differ from the composite route. The owner accepted a numerical
quality gate in place of bit-exact parity. Against the composite route, the step
has lower teacher-forced KL and higher top-1 agreement than the fused
recurrence that is already admitted for sampled serving. Passage NLL is no
higher on any tested passage. The carried recurrent state stays fp32 end to end.

## Operating it

- The NixOS option `hardware.tenstorrent.qwen38.gdnDecodeStep` defaults to `true`
  and sets `QWEN35_GDN_DECODE_FUSION=step`. Set it to `false` to return to the
  composite route.
- Greedy requests stay on the step route. Sampled requests keep the
  sampled-fusion route when `sampledFusion` is enabled. That kernel runs one
  (user, local value head) instance per core, so it fits 4 users of 24 heads on
  the 110-core P150. The batch worker caps an all-sampled batch at that width,
  and trace preparation warms the sampled route only at the widths it fits.
  Without `sampledFusion` the service accepts only temperature 0.
- The module rejects `mtpSpeculation` together with the step. There is no
  evidence for that combination.
- Unsupported geometry (non-128 heads, a batch above 32) keeps the composite
  route automatically.

## BF4 weights on this route

A screen on 2026-09-24 ran each existing BF4 gate on this route. The results are
recorded in `.cairn/changes/enable-qwen38-bf4-weight-caches/evidence/screen-2026-09-24.md`.

- `QWEN36_GDN_BF4=1` cuts the step by 5.9 percent (42.6 → 40.1 ms) and passes the
  declared accuracy floor. On two of eight prompts it skips the `<think>` block,
  so it stays default-off pending owner review.
- `QWEN36_MLP_DOWN_BF4=1` cuts the step by 5.3 percent but fails the NLL floor.
- `QWEN36_LM_HEAD_BF4=1` passes the floor but saves only 0.7 percent.

The NixOS option `gdnProjectionBf4` (default `false`) sets `QWEN36_GDN_BF4`.

## Host overhead and batch width

Decode used to block on a device synchronize after every trace replay. The
readback on the same command queue already orders after the replay, so that
synchronize is gone. Per-token time fell from about 46.4 to 45.1 ms, and the
greedy tokens stayed identical.

On this route one decode step costs 45.95 ms at width 1, 49.19 ms at width 8,
and 54.05 ms at width 16. Width 16 therefore produces 296 aggregate tokens per
second. Under concurrent load the packaged service keeps 50 to 58 ms per token
at widths 3 to 16. The composite route took 65 to 104 ms. The measurements are
in `.cairn/changes/serve-qwen38-fused-gdn-decode-step/evidence/host-and-width.md`.

## Prefill

Time to first token comes almost entirely from host-side TT-NN operation
dispatch, about 23,000 calls per prompt. Before these changes the service
prefilled a batch's users one after another. The fixes:

- Traced prefill (`tracedPrefill`, default `true`) replays one captured trace
  per 128-, 256-, 512-, or 1,024-token bucket, with the prompt length and decode
  slot passed as device data. Each replay installs the user's state and logits
  itself, so a batch's replays run back to back and read back once. Prompts
  longer than 1,024 tokens continue from the 1,024 bucket into a tail bucket.
  In a mixed batch, only the users no trace holds run eagerly. Output is
  identical to eager prefill. A 1,480-token prompt reaches its first token in
  0.8 s instead of 6–66 s. Measured through the packaged service, the worst
  time to first token was:
  - 2.2–2.6 s → 0.22 s at width 1;
  - 6.3–7.2 s → 1.69 s at width 8;
  - 10.8–17.6 s → 3.36 s at width 16.

  Aggregate throughput at width 16 rose from 33–64 to 115–120 tokens per second.
  The traces are captured at startup after every reachable program compiles,
  which adds 86–220 s. A later first-time compile releases them. See
  `.cairn/changes/trace-qwen38-masked-prefill/evidence/traced-prefill.md`.
- Grouped prefill (`groupedPrefill`, default `true`) runs a batch's users
  through shared forwards. GDN layers run batched, and full-attention layers
  loop per user. First-token logits and greedy text are identical to per-user
  prefill. Under concurrent load it cut the worst time to first token at width 4
  from 24.6 to 6.0 s, at width 8 from 30.1 to 15.8 s, and at width 16 from 53.3
  to 23.5 s. The per-user route remains for single users, prompts above 256
  tokens, and batches that reuse a parked decode trace. Traced prefill takes
  precedence, so grouped prefill serves only when the traces are off or
  released. See
  `.cairn/changes/qwen38-grouped-prefill-b8-rescreen/evidence/g7-exact-grouped-prefill.md`.
- Untraced prefill now splits prompts longer than 1,024 tokens into 1,024-token
  chunks with a masked tail. Before this change, prompts of 1,025 to 2,048
  tokens failed with an L1 out-of-memory error, or a circular-buffer clash in
  the MLP, beside the resident decode buffers. The composite route failed the
  same way.
- The fused chunk prefill op (`ttnn.transformer.chunk_gated_delta_rule`) sits
  behind `gdnFusedPrefill` / `QWEN36_GDN_FUSED_PREFILL=1`, default off. It cut
  short-prompt prefill from 2.2 to 1.1 s. Teacher-forced NLL rose by 0.007 and
  top-1 agreement with the composite prefill was 0.958. That misses the 0.97
  agreement bound, so the option stays opt-in. See
  `.cairn/changes/serve-qwen38-fused-gdn-decode-step/evidence/prefill.md`.

## Streamed greedy decode

The greedy trace tail combines the two shards' winners on the device. It writes
the tokens into the next step's input. A greedy batch then decodes with one step
queued ahead of the host's readback. Tokens are identical to step-by-step
decode. In the packaged service, the median step fell:

- from 45.9 to 42.5 ms at width 1;
- from 49.4 to 46.1 ms at width 8;
- from 55.0 to 51.4 ms at width 16.

At width 1 the step now equals the traced replay's device time. See
`.cairn/changes/stream-qwen38-greedy-decode/evidence/stream.md`.

## End of sequence

Generation stops at the model's end of turn. The stop ids come from
`generation_config.json` (`<|im_end|>`, `<|endoftext|>`). A batch stops once
every user has ended. Each response contains only the text before the stop id
and reports `finish_reason: "stop"`, or `"length"` when `max_tokens` was reached
first. Before this change every request ran to `max_tokens`, and invented user
and assistant turns followed the answer. With `max_tokens: 1024`, a 20-token
answer took 49.7 s; it now takes 1.7 s. Batch-admission and trace-reuse
admission measurements keep the fixed length. See
`.cairn/changes/stop-qwen38-at-end-of-sequence/evidence/chat-stop.md`.

## Non-claims

The measurements come from diagnostic sessions on a shared host whose load
average ran from about 40 to 95. Host-bound prefill timings varied by up to 3x
between runs. They are not an order-balanced admission receipt. They make no
claim for a deployed host. The production host configuration lives outside
this repository.
