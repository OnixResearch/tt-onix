# Head-normalization row candidate

This candidate kept normalized key vectors in head-row shape through WKV preparation.

The profiled B4 worker reduced tiled reshape launches from `5,187` to `4,251`. This is a reduction of `936`, or `18.045112781954888%`.

A five-pair alternating card-0 test compared the admitted transpose-fusion runtime with this candidate. Each sample processed 80 tokens and preserved the expected generated-token streams.

- Old aggregate throughput: `56.411431979659014` tokens/s.
- Candidate aggregate throughput: `52.17572636774648` tokens/s.
- Aggregate change: `-7.508594381081934%`.
- Mean throughput change: `-2.9411747756821316%`.

One pair contained long stalls in both variants. After that pair was excluded, the candidate gain was only `1.5654255791180072%`. Three of the other four pairs were slower.

The launch reduction did not produce a stable physical gain. The source change is reverted.

The A/B source run remains at `/var/tmp/rwkv7-p150x2-head-normalization-row-ab-20260809-1310/` on the measurement host.
