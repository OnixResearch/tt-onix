# Value-interpolation candidate

This candidate used `ttnn::lerp` for the token-local value anchor in later layers.

The profiled B4 worker replaced `1,287` binary launches with `429` ternary launches. This removes `858` launches.

A five-pair alternating card-0 test compared the admitted fused time-mix runtime with this candidate. Each sample processed 80 tokens and preserved the expected generated-token streams.

- Old aggregate throughput: `81.34426312894733` tokens/s.
- Candidate aggregate throughput: `64.48982555670668` tokens/s.
- Aggregate change: `-20.71988475145803%`.

One candidate sample contained a long stall. After that pair was excluded, the candidate gain was only `0.16619172135972082%`. Three of the other four pairs were slower.

The launch reduction did not produce a stable physical gain. The source change is reverted.

The A/B source run remains at `/var/tmp/rwkv7-p150x2-value-lerp-ab-20260809-1820/` on the measurement host.
