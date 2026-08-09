## Why

The runtime keeps only four live requests on each card. The model uses narrow 768-wide matrix products, so this small batch leaves much of each card idle. The current two-card record is `181.8205337965597` tokens per second. Reaching thousands of tokens per second requires more weight reuse and more work in each device program.

The next bounded step is a card-local capacity sweep. It must measure larger batches without weakening request isolation, FP32 recurrent state, exact token parity, latency bounds, or fail-closed evidence.

## What Changes

- Add typed candidate capacities for B4, B16, B32, and B64 card-local decode.
- Replace the fixed four-slot scheduler boundary with a pure profile-driven capacity plan.
- Allocate recurrent state from checked memory requirements before a device opens.
- Record effective batch size, queue occupancy, memory use, throughput, latency, and stalls.
- Admit a larger capacity only when simulator, parity, stability, and physical evidence pass.
- Retain the strongest measured blocker when no larger capacity passes.

## Impact

- **Runtime**: `pkgs/rwkv7-p150x2` gains profile-driven card-local capacity.
- **Evidence**: `pkgs/rwkv7-p150x2-evidence` gains capacity-sweep receipts and admission rules.
- **Configuration**: Nickel owns candidate capacities and resource bounds.
- **Testing**: positive and negative scheduler, memory, parity, and receipt cases are required.

## Out of Scope

- Changing model weights, precision, tokenizer, sampling, or correctness tolerances.
- Using a simulator result as physical throughput evidence.
- Claiming `1,000` tokens per second unless a complete physical receipt measures it.
- Adding more physical cards.
