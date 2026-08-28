# D1g/D1h: sampled serving implemented and validated (2026-08-28)

## Implementation (D1g)

- The sampled trace tail replaces the argmax winner reduction with a
  per-shard `ttnn.topk(k=20)` on the tile-layout sharded logits. One bug
  was found and fixed by the first device attempt: `topk` asserts
  `FillPad should only be used for tile layout` on row-major input, so the
  tail reduces the tile tensor directly.
- The exact-parity argmax tail is untouched for the off route; the two
  tails never mix because every mode switch releases all traces first.
- `decode_traced_sampled` replays the sampled trace and reads back each
  shard's top-K candidates with global token ids. The coordinator merges
  shards, applies the temperature softmax and nucleus cut, and draws with a
  per-request seeded generator (`seed` parses fail-closed). The first
  prefill token stays greedy argmax, matching the field convention.

## Physical validation (D1h)

Package `4bix76gd...` (identity `j9qxfhjw...`), private port, production
stopped and restored. Result: all checks pass
(`.pi/evidence/qwen38/sampled-serving-validation-20260828/result.json`).

- Greedy requests keep exact parity across mode switches and reproduce the
  known off-path tokens `[11751, 13, 198, 760]`.
- The first sampled request switches off->full; later sampled requests
  correctly need no switch; greedy after sampled switches back.
- `seed: 42` twice reproduces the identical 64-token stream.
- Sampled output differs from greedy (sampling is real) and stays coherent.
- Sampled decode 12.91 tok/s at B=1 versus 16.91 greedy: the top-K tail and
  candidate readback cost about 24 percent of the decode step, which the
  fused kernel's 1.064x partially offsets. This is the first real sampled
  serving on this deployment.

## Remaining

D1e only: deploy the generation with `QWEN36_SAMPLED_FUSION` still off,
then enable it as an explicit owner act. The deployed identity, receipt
bindings, and the archived B=4 admission stay valid for the off route.
