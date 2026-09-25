# Model-level speed and quality: composite, fused recurrence, fused step

Recorded 2026-09-24. Physical P150x2 run on devices `0` and `1`, B=1. The
runtime is `tt-metal-0.77.0`, and the checkpoint revision is
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. One process loaded the model once and
switched the GDN decode mode between runs (`off` composite, served;
`full` fused recurrence, admitted for sampled serving; `step` this change). The
throwaway probes are `.pi/tmp/gdn-step/model_eval.py` and
`.pi/tmp/gdn-step/timing_eval.py`. The raw log is `.pi/tmp/gdn-step/eval.log`.

## Speed

| Mode | Full decode trace replay (median of 5) | Per-token `decode_traced` (median of 55) | tok/s |
| --- | ---: | ---: | ---: |
| `off` | 54.33 ms | 59.04 ms | 16.9 |
| `full` | 50.90 ms | 55.33 ms | 18.1 |
| `step` | 42.69 ms | 45.55 ms | 22.0 |

The fused step removes 11.6 ms of device time from each token, a 21.4 percent
cut. It raises single-stream throughput by 1.30x against the served composite
route. Per-token host time is about 3 ms in every mode:

| Mode | binding check | input update | replay | sync | winner readback |
| --- | ---: | ---: | ---: | ---: | ---: |
| `off` | 0.84 | 2.08 | 54.32 | 0.74 | 1.06 |
| `step` | 0.80 | 1.27 | 42.51 | 0.23 | 0.75 |

## Quality gate

Teacher-forced runs feed the composite route's greedy tokens and compare full
logit rows position by position (6 prompts × 96 positions). Passage NLL is the
mean negative log-likelihood of three reference passages (prose, Python, and
history) after a 12-token prefix.

| Measure | `full` vs `off` | `step` vs `off` |
| --- | --- | --- |
| Teacher-forced top-1 agreement per prompt | 0.979 0.969 0.990 0.990 0.979 0.927 | 1.000 0.990 0.990 0.990 0.990 0.948 |
| Mean KL(off ‖ mode) per prompt | 2.1e-3 4.2e-3 1.7e-3 1.3e-3 2.4e-3 8.0e-3 | 1.5e-3 2.9e-3 7.8e-4 1.1e-3 2.3e-3 8.0e-3 |
| First greedy divergence index | 47 16 89 70 1 20 | none 77 89 70 1 1 |

| Passage NLL | prose | Python | history |
| --- | ---: | ---: | ---: |
| `off` | 0.8489 | 0.1311 | 1.0015 |
| `full` | 0.8447 | 0.1314 | 0.9990 |
| `step` | 0.8438 | 0.1295 | 0.9946 |

The step does not raise NLL on any passage. On every prompt, its logit distance
from the composite route is at or below the admitted fused recurrence. Both
fused routes diverge at token 1 on the TCP/UDP prompt, which is a near-tie at
the first position. The greedy texts stay coherent (see the log). Both gate
conditions in `r[tenstorrent.qwen38_p150x2.gdn_decode_step]` pass.

## Packaged service

The packaged serialized service,
`/nix/store/8pk8h84k6iivadhdk171dzqvnr0bmvwn-qwen38-0.1.0/bin/qwen38-p150x2-serve`,
ran with the module's arguments. Startup warm-up covered widths 1 and 4 on a
four-row batch buffer. The run then sent the 8 frozen greedy corpus requests
(`.pi/evidence/qwen38/b8-admission-20260826/request-0*.json`) sequentially,
twice per mode. The latency source is the service's own
`x_onix_decode_evidence.token_latency_seconds`, excluding each request's first
token. The runner is `.pi/tmp/gdn-step/service_ab.sh`.

| `QWEN35_GDN_DECODE_FUSION` | Token latencies | Median | tok/s | Serving route |
| --- | ---: | ---: | ---: | --- |
| `off` | 976 | 59.90 ms | 16.70 | `off` |
| `step` | 976 | 46.59 ms | 21.46 | `step` |

The service kept every greedy request on `step` with no fusion-mode switch. It
served the width-1 requests through the width-4 buffer, and the first 70
characters of every response matched the composite route.

## Non-claims

- This is a single-session diagnostic comparison, not an order-balanced
  admission receipt.
- There is no claim for B > 1 speed, long context, or sampled serving. The
  sampled route is unchanged.
- The production host configuration is outside this repository. This change
  does not claim a deployed service.
