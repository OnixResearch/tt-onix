# Qwen3.8 structured reads (`POST /v1/systemone`)

The P150x2 Qwen3.8-27B service answers Jev-style structured decisions. It serves
the contract that [vLLM PR 57250](https://github.com/vllm-project/vllm/pull/57250)
adds for DiffusionGemma. That PR was merged as `1b3b88ec2b7457aa030db4d0e7d8aaf04f6d0fb8`,
and its sources are under `.pi/references/vllm-pr57250`. The Cairn change is
`.cairn/changes/serve-qwen38-structured-reads`.

The PR itself cannot run on these cards:

- DiffusionGemma has no TT-NN port in this repository.
- The Tenstorrent vLLM plugin targets vLLM 0.26.

The owner chose the validated Qwen3.8 service instead. Qwen3.8 is autoregressive, so
the reads work differently. The request and answer shapes are the same.

## Request and answer

The body follows the PR's example server:

- `state` and `questions`. `questions` maps an id to `type` (`noul`, `choice`, or
  `score`), `instructions`, and `criteria`.
- The optional extensions `samples`, `auto_max`, `auto_threshold`, `steps`,
  `think`, `ask`, `chunk_rows`, `chunk_prompt`, and `sequential`, plus `depends_on`,
  `ask_if`, and `alone` on a question.

Answers take Jev's shapes:

- `noul` returns `{noul}`.
- `choice` returns `{choice, probabilities, confidence}`.
- `score` returns `{score, legend, probabilities, confidence}` with a 0-indexed legend.

A `diagnostics` block reports the stages, the chunks, the per-draw labels and
entropies, and the device read count and time.

```sh
curl -s localhost:8000/v1/systemone -H 'content-type: application/json' -d '{
  "state": {"ticket": "Everything is down and we have a demo at noon."},
  "questions": {"urgent": {"type": "noul", "instructions": "Does the customer need a reply within the hour?"}}}'
```

Differences from the PR:

- Images are refused with 422, because the service reads text only.
- `think` generates up to `--max-generation-tokens` greedy tokens with the chat
  template's low reasoning effort.
- `first_fill` (`placeholder` or `noise`) selects how the first draw fills the
  answer slots.
- `steps` defaults to 8, not 1. The first draw refines until its answers stop
  changing. In the measured corpus, that took one or two extra reads for most
  multi-question schemas and none for one-question schemas. `steps: 1` restores
  the PR's single read.
- The body has no `model` check.

Errors:

| Condition | Status |
| --- | --- |
| Invalid schema, multi-token label, cycle, over-long read, or images | 422 |
| Invalid JSON | 400 |
| Eager (MTP) mode, which has no batch worker | 503 |

## How a read works

The answer template, for example `urgent: yes\ntone: 1`, follows the chat prompt
as ordinary tokens. A slot's distribution is the next-token distribution at the
row before the slot. One prefill therefore reads every slot of a template. The
read stops before the last slot's own token, because no read row sees it.

Device side, in `pkgs/qwen36/structured-read-rows.patch`:

- **Wider logit tile.** Each masked prefill trace's one-hot logit selector widens
  from 1 to 32 rows.
  - Row 0 is still the generation prefill's next-token logit.
  - Rows 1–31 carry read positions, staged as device data like the prompt length.
  - The selector, final norm, LM head, and gather still cost one tile row.
  - Greedy generation is unchanged.
- **Batched read traces.** Two more traces, for the 128- and 256-token buckets,
  run four sequences in one forward. GDN, norm, and MLP layers run over all rows at
  once. Full-attention layers run per user through fixed page tables. The GDN path
  skips the state write-back, so the trace needs no masks.
- **Worker merging.** The batch worker merges queued decision jobs. Shortest reads
  go first into batched replays; a batched replay runs only where it beats
  singles and its captured KV write footprint does not overlap a retained
  generation prefix.
- **Prefix reuse.** A single read writes through its unchanged full-length
  page-table row; its first 16 KV blocks are reserved from generation. Batched
  reads capture four separate rows in blocks 4,096–4,111. A read whose plan
  starts with a full 1,024-token chunk saves the carried GDN state into
  persistent buffers. Those buffers are allocated, and their copy programs
  compiled, before any trace capture. A later
  read with the same first 1,024 tokens, and no read position inside them,
  restores that state and replays only its tail. Any prefill that writes one of
  the prefix's KV blocks drops the saved prefix first. Refinement steps and later
  draws of a long state therefore skip the shared chunk.
- **Eager fallback.** Without traces, reads run the same body eagerly, with no
  prefix reuse.
- **Generation cache isolation.** Traced single reads report the KV blocks
  through their padded final bucket. Batched traces report every captured row,
  including unused lanes. The continuous-slot worker retains generation
  prefixes whose physical KV ranges are disjoint from those blocks and evicts
  overlapping entries; eager reads, lost masked traces, and failures retain the
  conservative full invalidation. A physical 800-token read preserved a
  130,048-token generation prefix (1.084 s continuation TTFT versus 103.696 s
  before); a 1,400-token read overlapped it and forced a cold 103.953 s
  continuation. See the [paired receipt](../pkgs/qwen36/evidence/2026-09-30/qwen38-structured-read-prefix-retention.json).
- **Dedicated batched-read KV.** The 4,113-block candidate reserves four
  disjoint four-block rows immediately after generation's blocks 16–4,095,
  followed by scratch block 4,112. Both 128- and 256-token replays use these
  fixed rows, including unused lanes; entries past the four-block runs point
  to scratch. Short batched reads therefore cannot overlap generation KV.
  Longer single reads retain the full-length row from block 0 and can still
  overwrite generation KV beyond their reserved first 1,024 tokens.
  A [bounded physical receipt](../pkgs/qwen36/evidence/2026-09-30/qwen38-dedicated-read-batch-lanes-physical.json)
  observed three batched routes among four concurrent 82-token reads in the
  128 bucket and three among four 198-token reads in the 256 bucket. Both
  bursts retained A's 130,048-token and B's 16,384-token generation prefixes
  and preserved generated token IDs. This does not measure longer reads or
  establish general throughput.
  A historical [y58 paired root-systemd receipt](../pkgs/qwen36/evidence/2026-09-30/qwen38-read-batch-lanes-systemd-paired.json)
  ran the first built 4,113-block unit with identical six-request SHA-256 inputs
  against the old package. Both 82-token and 198-token bursts took three
  batched traces plus one masked trace and returned the same numeric answers
  as baseline; A and B generated the same token IDs. The candidate kept
  130,048- and 16,384-token prefix hits through both bursts. The old
  2,048-cached-block package lost B's 16,384-token hit after the short burst;
  the candidate also raises that budget to 4,064, so continuation TTFT
  differences cannot be ascribed to the read lanes alone. That root unit
  passed notify readiness, but its temporary override was removed and
  the old package restored. This is bounded evidence for y58, not broad
  throughput or production admission for another package output.
  The [committed e3cfded han2 physical receipt](../pkgs/qwen36/evidence/2026-09-30/qwen38-han2-committed-read-lanes-physical.json)
  separately exercised its changed package output under root-owned
  `Type=notify`, ready after 622 s. It replayed the same six exact request
  hashes and matched the **preserved**, not freshly rerun, old and y58
  generated-token IDs and numeric read answers. Both four-way 82- and
  198-token bursts took three batched routes and one masked route and
  retained A's 130,048-token and B's 16,384-token prefixes before and after
  each burst. The temporary candidate stopped and the original `gfvkac4`
  root service returned ready with unchanged 90/95/96 drop-ins. This is
  bounded requalification of han2, not a durable host-generation switch.
  The preceding [4,097-block candidate](../pkgs/qwen36/evidence/2026-09-30/qwen38-cache-aware-batched-read-physical.json)
  instead used startup generation rows for batches and routed overlapping
  batches to single-read traces. Its four 84-token decisions preserved a
  16,384-token prefix (0.414 s warm TTFT versus 9.598 s cold), but three
  replies took about 0.90 s instead of 0.64 s. Those are historical
  measurements, not measurements of the dedicated-lane layout.

Policy side, in `structured_read_core.py`. The PR's policy is noise in every
slot, one denoise step, and draws differing only in noise. That fits a diffusion
canvas, not an autoregressive model:

- **Refined first draw.** The first draw fills slots with a `?` placeholder and
  refines to a fixed point. By default (`steps` up to 8), each step refills the
  slots with the previous argmax labels, until the fills stop changing. That fixed
  point is the greedy answer chain.
- **Sampled later draws.** Each later draw reads once, with earlier slots filled
  by labels sampled from the first draw.
- **Auto policy.** The auto policy ignores the first slot's entropy, because the
  row before the first slot sees no earlier slot.
- **Memoization.** Identical prefills reach the device once per decision. A
  one-question decision is one read for any `samples` or `steps` value.

## Measured on P150x2 (2026-09-25/26)

All numbers come from the packaged service or a probe on the same package. The
host was shared.

### Correctness

- **Read rows.** Every read row equals the logits of a prefill ending at that
  row: max |Δ log p| is 0.0 over the top 20 on all 9 rows of a 137-token prompt.
  Row T−1 equals the generation prefill's installed logit bit for bit.
- **Batched reads.** Batched reads equal the same sequences read alone bit for
  bit (`torch.equal`), for lengths 51–161. Eager-fallback reads equal traced reads
  bit for bit.
- **Prefix reuse.** A read that reuses a saved 1,024-token prefix equals the
  uncached read bit for bit. This held after a width-1 generation and after
  batched reads. A short read through the same row, or a width-16 prefill, drops
  the prefix, and the next read replays from position 0.
- **Generation.** Greedy generation is token-identical to the package without
  this change: 5 prompts of up to 48 tokens each, checked before and after
  structured reads ran.
- **Mixed load.** 12 mixed concurrent chat and structured requests all returned
  200. Error paths return the statuses above.

### Latency

Single read replay by prompt length:

| Tokens | 128 | 200 | 300 | 512 | 1024 | 1500 | 2000 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Read (s) | 0.23 | 0.26 | 0.36 | 0.36 | 0.56 | 0.90 | 1.11 |

The eager fallback takes about 2.0–2.2 s.

Batched replay against single replays:

| Bucket | 1 sequence | 4 sequences, batched | 4 sequences, single |
| --- | ---: | ---: | ---: |
| 128 | 0.42 s | 0.42 s | 0.90 s |
| 256 | 0.68 s | 0.68 s | 1.01 s |

The planner batches a group only where it wins: at least 2 sequences in the
128-token bucket and at least 3 in the 256-token bucket.

A read past 1,024 tokens that shares its first chunk with the previous read on
its row replays only its tail:

| Read tokens | Uncached | Reused prefix |
| ---: | ---: | ---: |
| 1,431 | 0.89 s | 0.36 s |
| 1,813 | 1.11 s | 0.57 s |

Saving the prefix adds no measurable time to the first read.

### Accuracy and throughput

The test corpus has 48 decisions and 132 labelled questions:

- programming language (10);
- human language (10);
- unit comparison (8);
- primality (6);
- support tickets with 3 questions each (8);
- one 10-question parity schema;
- one 40-question parity schema;
- 6-question item surveys (4).

| Policy | Accuracy | p50 latency | Decisions/s |
| --- | ---: | ---: | ---: |
| PR-style: noise, 1 step, auto | 121/132 | 0.26 s | 2.14 |
| placeholder, 1 step, auto | 124/132 | 0.26 s | 2.08 |
| noise, refine ≤ 8, auto | 128/132 | 0.26 s | 2.08 |
| **default: placeholder, refine ≤ 8, auto** | **128/132** | 0.26 s | 2.16 |
| default, `samples: 1` | 128/132 | 0.26 s | 2.35 |
| default, 8 concurrent clients | 128/132 | 1.89 s | 2.50 |
| default, `samples: 1`, 8 concurrent clients | 128/132 | 1.90 s | 2.82 |

Refinement fixes the 40-question schema, from 29–36/40 to 40/40, whichever
first fill starts it. It costs 1.45–1.57 device reads per decision on average.

The auto policy with the first-slot gate and memoization cut device reads from
87 to 66 on the 42-decision subset. Accuracy stayed the same, and single-stream
throughput rose from 1.9 to 2.6 decisions/s.

Batched read traces raised auto throughput at 8 concurrent clients from 2.5 to 3.4
decisions/s. This was measured with the one-step policy on the 42-decision subset.

Prefix reuse was measured on four long support threads, with 1,265–1,533 prompt
tokens and three questions each:

- p50 latency fell from 2.02 to 1.35 s under the default policy;
- throughput rose from 0.48 to 0.70 decisions/s;
- 12/12 were correct either way;
- a three-read decision fell from 2.43 to 1.36 s.

On the 48-decision corpus, whose prompts stay under 1,024 tokens, the results
were unchanged: 128/132 at 2.14 decisions/s.

## Cost and limits

- **Startup.** The two batched read traces add about 3 s to trace capture. The
  saved prefix state takes about 87 MiB of DRAM per device, computed from the
  shapes: 48 GDN layers' fp32 recurrent state plus conv carry.
- **Service time.** Reads run on the batch worker between generation batches. A
  decision waits for any running generation batch, and generation waits for
  queued reads.
- **Sequence length.** A single read may use up to the configured
  `--max-sequence-length` (131,072 tokens in the updated module and the running
  runtime-overridden service; the unchanged NixOS generation reverts to 65,536
  after reboot). Its first 1,024 tokens use blocks 0–15, which no generation
  group receives; a longer single read can write generation blocks and
  invalidate overlapping cached prefixes. In the undeployed 4,113-block
  read-lane candidate, four short batched reads instead use separate blocks
  4,096–4,111 outside the 4,080-block generation range, with block 4,112 as
  scratch. The direct, historical y58 and committed han2 physical receipts
  cover 82- and 198-token reads, not a longer single read; han2 remains
  undeployed. The first direct attempt omitted production grouped and traced
  prefill flags and did not measure a batched read route.
  One read holds at most 31 slots. Larger schemas are chunked, as in the PR.
  The measured read corpus uses short prompts; 128K-token structured-read
  quality is untested.
- **Disk.** Every rebuilt package compiles a new tt-metal kernel cache, about
  2.2 GB per package, under `<model>/.cache/tt-metal-cache`. Remove stale entries
  when the disk fills. A full disk fails the trace capture with
  `masked-prefill-traces-failed`.

## Non-claims

- The corpus sessions are diagnostic; the historical y58 and committed han2
  root-systemd receipts cover only their exact A/B and cached-read fixture,
  not general throughput, broad quality or TT-Metal 0.77 production admission.
- The corpus is small and hand-labelled.
- Qwen3.8's accuracy is not comparable to the DiffusionGemma numbers in the PR.
- The shared host's accuracy and latency may differ from a deployed host.
