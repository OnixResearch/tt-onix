# Capacity inventory, implementation, and B=8 scheduler smoke

Recorded 2026-10-21. Package `39p6vlfkf15jy77lh9x8k2gzxqp7fnwx-qwen38-0.1.0`, execution
identity `7bxzfg0q0hfkikriygh9wi049mdn7fbd-qwen36-execution-identity-0.1.0`, runtime
`7hplkynh047dsk98cwqpfsaaq6lwyy0m-tt-metal-0.77.0`, physical PCIe devices 0 and 1.

## A1: the three capacity pins

| Site | Before | After |
| --- | --- | --- |
| `continuous_slot_core.SLOT_CAPACITY` | constant `4`, policy equality check | `SUPPORTED_SLOT_CAPACITIES = (4, 8)`, membership check |
| `continuous-slot-contracts.ncl` | `slot_capacity == 4` | unchanged (B4); sibling `-b8.ncl` pins `== 8` |
| `continuous-slot-policy.ncl` | `slot_capacity = 4` | unchanged (served default); sibling `-b8.ncl` declares 8 |

The parse path now bounds the integer at `MAXIMUM_SLOT_CAPACITY = 8` and then applies the
closed-set membership check, so 1, 5, 6, and 16 are refused before any device work.

## A2: the coordinator is capacity-generic

`ContinuousSlotCoordinator` opens `begin_slots(width=self.slot_policy.slot_capacity)` and
sizes the inactive rows, the decode window, and the reset from the same value. Its only
capacity-specific text was the module docstring, which now says "the profile's decode
slots". The adapter side already supports width 8: the B=8 admission binary captures and
replays a width-8 trace, and `maximum_width` is 16.

## I1-I4: implementation

- Closed-set capacity in the pure core, with the parse bound and the membership error.
- New `continuous-slot-contracts-b8.ncl` and `continuous-slot-policy-b8.ncl`: identical
  queue, decode-window, refill, bypass, deadline, and stall bounds; `slot_capacity = 8`;
  profile id `qwen38-p150x2-continuous-slot-b8-v1`. The B4 files are byte-identical.
- The package exports `continuous-slot-policy-b8.json` beside the B4 JSON and binds both
  policy files into the execution identity. The served default is unchanged.
- Coordinator wording generalized.

## V1-V3: device-free

The package build and its install check pass. The policy probe now:

- accepts the B4 and B8 policies from their exported JSON,
- requires the B8 profile id and exactly eight slots from `initial_schedule`,
- requires every non-capacity field of B8 to equal B4,
- accepts capacity 8 through the mapping constructor,
- refuses capacities 1, 5, 6, and 16 in addition to the existing malformed-field cases.

## V4: physical B=8 scheduler smoke

The production `qwen38-p150x2-serve` binary ran with
`--continuous-slot-policy …/continuous-slot-policy-b8.json` and the package prefix-cache
policy. Two barrier waves, temperature 0:

| Wave | Requests | Usable | Wall | Path |
| ---: | ---: | ---: | ---: | --- |
| 4 | 4 | 4/4 | 6.45 s | `continuous-slot` |
| 8 | 8 | 8/8 | 7.65 s | `continuous-slot` |

Every response was HTTP 200. The four prompts shared by both waves produced identical
text at width 4 and width 8 (the printed continuations match), so the wider group did not
change a user's output. Prompts that hit a stop token finished early (8 and 11 tokens);
the others generated the full 64.

## Non-claims

- This is a feasibility smoke, not an admission receipt. The B=8 profile remains a
  candidate; V5 (the queue-inclusive admission comparison with the B=8 workload) is open.
- B4 remains the served default. No NixOS default, production policy, or deployment
  changed.
- The smoke does not establish throughput, tail latency, fairness, or long-prompt
  behavior at eight slots; those are V5's gates.
- The production response evidence does not carry the admission-only `batch_width` and
  `state_slice` fields; route and output are the checked properties here.
