# Device-continuation investigation

A temporary B1 path kept selected token tensors on the device between model steps. It read the first and final selected tokens during timing. It read intermediate token evidence after device completion.

All tested B1 runs produced the same 64-token stream as the blocking-readback baseline.

An initial five-repeat run increased aggregate throughput from `9.31519634811253` to `11.32439245891554` tokens/s. Later samples showed much larger machine-time variation.

A five-pair alternating A/B test gave the strongest comparison:

- Blocking-readback aggregate: `22.30764499564398` tokens/s.
- Device-continuation aggregate: `20.673600551461266` tokens/s.
- Aggregate change: `-7.325042354322009%`.
- Mean rate change: `0.11850828141355407%`.
- Median rate change: `14.030194778215389%`.

The device-continuation path was faster in four pairs. One `5.20511783`-second sample removed the aggregate gain. This evidence does not prove a sustained improvement.

Dual-worker profiler validation also began returning incorrect card 0 tokens. The previously validated package reproduced this failure before and after a board reset. Unprofiled isolated and concurrent workers retained exact tokens.

The experiment was reverted. Do not claim a device-continuation gain until dual profiler parity and machine-time stability return.

Raw B1 A/B receipts remain under `/var/tmp/rwkv7-p150x2-b1-alternating-ab-20260808-2200/` on the measurement host.
