## Why

Program caching removes repeated compilation, but the host still submits many programs for every generated token. At thousands of tokens per second, the allowed time per token becomes too small for repeated host scheduling and synchronization.

The runtime needs a bounded device-resident decode mode. It should keep token IDs, recurrent state, and decode control on the card across several steps. The host should intervene only at declared request boundaries, cancellation, errors, or receipt collection.

## What Changes

- Add a typed persistent-decode contract with bounded step and time limits.
- Validate dynamic token, state, slot, and output bindings before replay or persistence.
- Keep next-token selection and recurrent-state carry on the selected card.
- Add deterministic completion, cancellation, timeout, and error handshakes.
- Retain normal dispatch as a fail-closed fallback.
- Require exact parity, simulator smoke, profiler evidence, and stable physical admission.

## Impact

- **Runtime**: `pkgs/rwkv7-p150x2` gains an optional persistent decode shell.
- **Scheduling**: request lifecycle gains explicit device-active, completed, cancelled, and failed states.
- **Evidence**: receipts report host submissions per token, device step counts, exits, and binding identities.
- **Testing**: stale-binding, slot-reuse, timeout, cancellation, completion, and parity cases are required.

## Out of Scope

- An unbounded device loop.
- Silent fallback after a binding or parity failure.
- Changing greedy token selection or sampling behavior.
- Claiming speed from reduced host calls alone.
- Removing the existing evidence and normal-dispatch paths.
