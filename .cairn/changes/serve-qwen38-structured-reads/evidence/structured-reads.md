# Structured reads on P150x2: evidence (2026-09-25/26)

These are diagnostic sessions on a shared host, not an admission receipt. The model is
Qwen3.8-27B at checkpoint `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, with physical devices 0 and 1. The environment
matched the NixOS defaults: `QWEN35_GDN_DECODE_FUSION=step`, grouped and traced prefill on, and
sampled fusion and MTP off. All runs used `--max-sequence-length 2048` and
`--max-generation-tokens 64`.

## Primitive (probe on the patched package)

- **Read rows against independent prefills.** The prompt had 137 tokens plus an
  8-token answer canvas. There were 9 read rows, for positions 136–144.
  - Each row's log-probabilities equal a separate prefill ending at that row:
    max |Δ| is 0.0 over the top 20, and the top-20 overlap is 20/20.
  - Row T−1 equals the generation prefill's installed slot logit (max |Δ| 0.0).
- **Batched against single.** Batched reads were compared with the same
  sequences read alone.
  - Four sequences of 51, 52, 53, and 161 tokens used the batched trace. Each
    equals its single read (`torch.equal`).
  - Sequences of 47 and 301 tokens took the single masked trace, as planned.
  - Eager-fallback reads, with the traces released, equal traced reads
    (`torch.equal`).
- **Greedy tokens.** Greedy tokens before and after all reads were identical,
  and the traces stayed fresh (`stale=false`).
- **Single read replay latency.**

  | Tokens | Read (s) |
  | ---: | ---: |
  | 100 | 0.23 |
  | 128 | 0.23 |
  | 200 | 0.26 |
  | 300 | 0.36 |
  | 512 | 0.35–0.38 |
  | 700 | 0.56–0.67 |
  | 1024 | 0.56 |
  | 1100 | 0.77 |
  | 1500 | 0.90 |
  | 2000 | 1.11 |

  The eager fallback took 1.99–2.25 s.
- **Batched replay against single replays.** The best of 3 runs:

  | Bucket | 1 seq batched | 1 seq single | 2 seq batched | 2 seq single | 3 seq batched | 3 seq single | 4 seq batched | 4 seq single |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | 128 | 0.418 | 0.222 | 0.418 | 0.441 | 0.419 | 0.675 | 0.420 | 0.899 |
  | 256 | 0.683 | 0.252 | 0.683 | 0.505 | 0.684 | 0.766 | 0.682 | 1.014 |

  Minimum batch sizes follow from these numbers: 2 for the 128 bucket and 3 for
  the 256 bucket.
- **Failed capture order.** The first batched-trace build captured the batched
  body after the single-sequence traces, and a batched read then hung the device.
  The cards were recovered with `tt-smi -r`. Every body now compiles before the
  first capture, and the following runs passed.
- **Prefix reuse** (package `n4jyna84…`, 2026-09-26). A 1,807-token prompt with
  a 6-token template made an 1,813-token read, planned as a 1,024-token chunk plus
  a tail. Single reads use page-table row 15.
  - The first read saved the prefix at no extra cost: 1.105 s, the same as an
    uncached read (1.105 s).
  - A read with other template fills reused the prefix in 0.571 s. Its logits
    equal the uncached read of the same tokens (`torch.equal`).
  - It still reused, and still matched, after a width-1 generation (row 0) and
    after three batched reads (rows 0–3).
  - A short single read through row 15 dropped the prefix, and so did a width-16
    prefill (rows 0–15). The next read replayed from position 0 and matched.
  - Best of 3, uncached against reused:

    | Read tokens | Uncached | Reused |
    | ---: | ---: | ---: |
    | 1,431 | 0.893 s | 0.356 s |
    | 1,813 | 1.111 s | 0.574 s |

  - Greedy tokens were unchanged afterwards, and the traces stayed fresh.

## Service (packaged `qwen38-p150x2-serve`)

- **Generation parity.** Five chat prompts with up to 48 greedy tokens each were
  token-identical to the package at the previous commit
  (`ffbh5mg4…-qwen38-0.1.0`). This was checked on five candidate packages,
  including the final one (`1dy7hmzi…`).
- **Corpus.** The corpus has 48 decisions and 132 questions. The results for each
  policy:

  | Policy | Accuracy | p50 latency | Decisions/s | Device reads |
  | --- | ---: | ---: | ---: | ---: |
  | noise, 1 step, auto (PR-style) | 121/132 | 0.26 s | 2.14 | 79 |
  | placeholder, 1 step, auto | 124/132 | 0.26 s | 2.08 | 82 |
  | placeholder, 1 step, `samples: 1` | 110/132 | 0.26 s | 3.69 | 49 |
  | noise, 1 step, `samples: 1` | 116/132 | 0.26 s | 3.67 | 49 |
  | noise, refine ≤ 8, auto | 128/132 | 0.26 s | 2.08 | 78 |
  | **placeholder, refine ≤ 8, auto (default)** | **128/132** | 0.26 s | 2.16 | 77 |
  | default, `samples: 1` | 128/132 | 0.26 s | 2.35 | 71 |
  | default, 8 clients | 128/132 | 1.89 s | 2.50 | 77 |
  | default, 16 clients | 128/132 | 3.29 s | 2.76 | 77 |
  | default, `samples: 1`, 8 clients | 128/132 | 1.90 s | 2.82 | 71 |

  - Before the first-slot gate and memoization, on the 42-decision subset:
    55/58, 129 reads, 87 device reads, 1.89 decisions/s.
  - After them: 55/58, 66 reads, 66 device reads, 2.48 decisions/s.
  - Batched read traces, with the 1-step policy on the same subset, raised
    throughput at 8 clients from 2.52 to 3.35 decisions/s.
- **Long states** (package `1dy7hmzi…`, with prefix reuse, against `bdq7flg8…`,
  without). Four support threads of 1,265–1,533 prompt tokens, three questions
  each:

  | Policy | Accuracy | p50 latency | Decisions/s | Device reads |
  | --- | ---: | ---: | ---: | ---: |
  | auto, without reuse | 12/12 | 2.02 s | 0.48 | 9 |
  | auto, with reuse | 12/12 | 1.35 s | 0.70 | 9 |
  | `samples: 4`, without reuse | 12/12 | 2.02 s | 0.48 | 9 |
  | `samples: 4`, with reuse | 12/12 | 1.32 s | 0.71 | 9 |

  - Each decision's first read took the masked trace, and its later reads reused
    the prefix.
  - Per decision: 1.82 → 1.31 s, 2.23 → 1.70 s, 1.82 → 1.33 s, and 2.43 → 1.36 s
    (the three-read one).
  - On the 48-decision corpus, whose prompts stay under 1,024 tokens, the same
    package was unchanged: auto 128/132 at 2.14 decisions/s, and `samples: 1`
    128/132 at 2.34.
  - Greedy tokens were still identical to the previous commit's package.
- **Contract checks.** All ran against the live service, and they were re-run
  with the same results on the final package (`1dy7hmzi…`). That package's full
  corpus also held at 8 and 16 clients: 128/132 at 2.52 and 2.59 decisions/s.
  - `think: 32` wrote a 32-token thought (3.3 s total).
  - `steps: 3`, `depends_on` with `ask_if` skipping a question,
    `sequential: true` with `chunk_rows: 8`, and `ask` all returned 200.
  - A 40-question schema read as chunks of 31 and 9 slots and answered 40/40
    correctly in 4.7 s.
  - A thought over budget, images, an unknown type, and 27 alternatives returned
    422. Invalid JSON returned 400.
  - 12 concurrent mixed chat and `/v1/systemone` requests all returned 200.
- **Disk.** Each rebuilt package compiled a new kernel cache of about 2.2 GB. Ten
  candidate builds filled the disk, and trace capture then failed with
  `No space left on device`. Removing this session's cache
  (`14735689022491282755`) and the intermediate store paths recovered the service.
