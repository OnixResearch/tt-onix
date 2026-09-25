## Why

A traced B=1 decode step runs for 42.5 ms on the device. Each step through the
batch worker took 45.6 ms. The difference was host work on the critical path:

- two blocking readbacks of per-shard indices and values (1.2 ms);
- the host-side combine of the two shards' winners;
- staging the next token;
- the idle time before the next replay.

The device waited for the host on every token.

## What Changes

- The greedy trace tail combines the two shards' winners on the device. It
  all-gathers each shard's maximum and local index, keeping device 0 on ties as
  before. It then writes the global tokens into the next step's token input.
- A greedy batch decodes through `decode_greedy_stream`. Only the first step
  stages tokens from the host; every step stages its positions and RoPE rows.
  The host reads step k's tokens without blocking while step k + 1 runs.
- The greedy readback reads the combined tokens from one replica, in one
  transfer.
- Sampled batches keep their per-step host sampling.

## Impact

- **Files**:
  - `pkgs/qwen36/service/qwen36_service/{batch_decode_adapter,batch_service}.py`;
  - `pkgs/qwen36/warmup-tail-check.py`;
  - `pkgs/qwen36/readback-probe.py`, removed with the two-read readback it
    measured;
  - `docs/qwen-readback-evaluation.md`.
- **Behavior**: greedy tokens are unchanged. After an end-of-sequence stop or a
  timeout, one queued step is computed and dropped.
