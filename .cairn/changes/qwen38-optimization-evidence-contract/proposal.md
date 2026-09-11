## Why

Optimization decisions need a current, content-bound baseline. The adoption document mixes historical eager results with later traced and sampled results. A renamed checkpoint also concealed an MTP precision change.

The September research report records these conflicts in `docs/research/qwen38-p150x2-2026-09-04.md`.

## What Changes

- Bind backbone conversion, BF16 MTP head, tokenizer, cache format, runtime, route, and hardware scope in one experiment manifest.
- Separate diagnostic observations from admitted performance.
- Define complete round, request-wall, queue, prefill, and decode timing.
- Add a bounded experiment shell with exclusive device ownership and explicit outage authority.

## Impact

`pkgs/qwen36` owns experiment meaning. Existing receipt cores and Rust receipt tools remain the starting components. Typed Nickel owns profiles. TTNN, process control, clocks, and filesystem work remain in adapters and shells.

Positive and negative fixtures must reject stale identities and incomplete roots. No current admission transfers automatically. This proposal authorizes no hardware run or service interruption.
