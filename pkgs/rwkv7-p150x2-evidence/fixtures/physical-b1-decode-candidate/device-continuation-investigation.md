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

Further isolation changed the dual-worker diagnosis. Card 0 kept exact tokens when it ran alone, with or without profiling. Card 1 also kept exact tokens.

Concurrent independent workers could corrupt card 0, with or without profiling. Some concurrent runs passed, so the failure is timing-dependent. The previously validated package reproduced the same failure.

Board resets and a three-second settle interval did not reliably restore parity. The cause was missing process isolation, not a profiler-only defect.

Each worker used the dual-card descriptor. Both UMD processes therefore discovered and opened both physical cards.

A later fix filtered each process with `TT_VISIBLE_DEVICES`. Each process now uses the single-card P150 descriptor and opens logical device 0.

Five concurrent control samples and one complete strict benchmark kept exact tokens. This resolves the dual-worker parity blocker.

The continuation experiment remains reverted. Do not claim a device-continuation gain without a stable throughput improvement.

Raw B1 A/B receipts remain under `/var/tmp/rwkv7-p150x2-b1-alternating-ab-20260808-2200/` on the measurement host.

Profiler-isolation receipts remain under `/var/tmp/rwkv7-p150x2-profiler-isolation-card0-profiled-20260808-2240/`. Reset-control receipts remain under `/var/tmp/rwkv7-p150x2-reset-settle-control-20260809-0040/`.
