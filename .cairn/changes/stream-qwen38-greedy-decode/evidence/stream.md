# Streamed greedy decode on P150x2

Recorded 2026-09-25 on devices 0 and 1, runtime `tt-metal-0.77.0`, with the
fused GDN decode step. The throwaway harnesses are `.pi/tmp/gdn-step/`:
`test_feedback.py`, `stream_eval.py`, `stream_iso.py`, and `service_stream.sh`.
The host was shared, with a load average between 23 and 67.

## Where the host time went

At width 1 the traced replay took 42.50 ms and a full step through the adapter
took 45.56 ms. Host input preparation took 0.08 ms. The two blocking shard
readbacks with the host combine took 1.24 ms. The rest was the device idling
between the readback and the next replay.

## Device winner combine

The combine was checked on the mesh with vocabulary shards of 124,160 at widths
1, 5, and 16, in three trials each:

- random logits;
- equal maxima placed on both shards (device 0 must win);
- a strictly larger maximum on device 1.

In all 9 cases the device tokens equalled the host combine and torch's argmax
over the full vocabulary, and the two replicas agreed. Reading the combined
tokens from one replica took 0.149 ms. The old two-read host combine took
1.173 ms.

## Streamed, step-by-step, and previous decode

Each run did 64 greedy steps after prefill, in separate processes. The
reference is the previous adapter.

| Width | Previous adapter | New, step by step | New, streamed |
| ---: | ---: | ---: | ---: |
| 1 | 46.56 ms | 45.13 ms | 42.59 ms |
| 8 | 49.33 ms | 48.88 ms | 46.17 ms |
| 16 | 54.83 ms | 53.95 ms | 51.39 ms |

The tokens of the 65 steps (first token plus 64) equalled the previous adapter
at every width, both step by step and streamed.

## Packaged service

Two packages were compared under concurrent greedy load: the previous package
`/nix/store/vd0k32r0z748rz0l1rx7isrbi2v32k1a-qwen38-0.1.0` and
`/nix/store/ffbh5mg4wig95m5ip9k8x53s0j18jhla-qwen38-0.1.0`. Each request asked
for 256 tokens. Each width ran two rounds, and each round decoded 254 steps.

| Width | Median step, before | Median step, after | p99 step, before | p99 step, after |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 45.8–45.9 ms | 42.5–42.6 ms | 50.9–52.0 ms | 43.4–45.3 ms |
| 4 | 47.0 ms | 43.9 ms | 51.8–52.9 ms | 46.7–51.5 ms |
| 8 | 49.2–49.5 ms | 46.1–46.2 ms | 53.7–54.2 ms | 51.0–51.6 ms |
| 16 | 54.9–55.1 ms | 51.4–51.5 ms | 59.9–60.4 ms | 56.3–58.2 ms |

- Generated token ids were identical in 58 of 58 responses.
- Summed decode time per request fell by 6 to 8 percent, for example from
  11.7 to 10.8 s at width 1 and from 14.0 to 13.1 s at width 16.

## Per-request trace capture is load-bound

In the service run the capture took longer on the new package. The load
average had risen to 67 by then. An isolation run in one process then
interleaved step-by-step and streamed requests at a load average of about 55.
The next request's capture took:

- after step-by-step decode, 2.8 to 4.8 s;
- after streamed decode, 2.0 to 3.1 s;
- after streamed decode with one read in flight, 1.9 to 4.2 s.

So the capture cost follows host load, not the decode mode. It is dispatched
from Python like an eager forward.
