## Context

Before this change, the greedy tail left `(indices, values)` per shard on the
devices. `_read_global_winners` read both through two blocking `to_torch` calls,
each covering both devices, and picked the winner with `torch.argmax` over
devices. The next `decode_traced` call then staged that token, and it could only
start after the readback.

## Decisions

### Decision: Combine on the device, in float32

**Choice:** `_combine_shard_winners`:

1. typecasts each shard's index and maximum to float32;
2. all-gathers the pair along the device dimension;
3. computes `index0 + (value1 > value0) * (index1 + shard - index0)`;
4. converts the result back to `uint32 [width, 1]`.

**Rationale:** A token id is below 2**24, so float32 holds it exactly.
`value1 > value0` keeps device 0 on equal maxima, as `torch.argmax` did. A device
test compared the result with the host combine and with torch's full-vocabulary
argmax for widths 1, 5, and 16, including forced ties and cross-shard wins.
Every case matched, and both replicas agreed.

### Decision: Feed the winners back inside the trace

**Choice:** The trace ends with
`ttnn.copy(winners, device_inputs[0])`. `decode_traced` still stages the token
from the host, overwriting it with the same value. `decode_greedy_stream`
stages the token only for its first step.

**Rationale:** The next replay then depends on nothing the host has to read
first.

### Decision: One step in flight

**Choice:** Each stream step:

1. stages positions and RoPE;
2. replays;
3. queues `ttnn.from_device(replica, blocking=False)` and records an event.

The host waits for the previous step's event before yielding its tokens.
Closing the stream waits for every queued step.

**Rationale:** All reads, writes, and replays share one command queue, so each
read lands between its own replay and the next. Staging for step k + 1 lands
after replay k has consumed its inputs. One queued step hides the host work.
The cost at an early stop is one dropped step.
