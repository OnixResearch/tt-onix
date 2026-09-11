# Physical B1 decode candidate

This candidate measured one physical P150 card with one active request. The prompt was token `0`, and each run generated 64 tokens.

Five repeats produced the same generated-token stream. Program-cache entries stayed stable during each timed region.

- Mean throughput: `10.318388240286174` tokens/s.
- Aggregate throughput: `9.31519634811253` tokens/s.
- Median throughput: `8.087155632081794` tokens/s.
- Range: `6.675900729440436` to `14.994682237861321` tokens/s.
- Maximum latency: `9.586721342` seconds.

Card 0 ended at `800 MHz` and `66.4°C`. This result does not show thermal throttling.

The wide throughput range shows nonstationary latency. This candidate has no profiler attribution or pinned FP32 CPU-reference comparison. It cannot satisfy physical scaling admission.

Raw receipts remain under `/var/tmp/rwkv7-p150x2-b1-decode-repeats-20260808-1430/` on the measurement host.
