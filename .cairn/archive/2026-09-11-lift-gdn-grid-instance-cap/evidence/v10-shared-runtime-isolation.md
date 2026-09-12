# V10: the shared tt-metal output is untouched and its consumers still pass

Recorded 2026-09-11 for change `lift-gdn-grid-instance-cap`, task V10.

## Shared output identity

```
nix build .#tt-metal --no-link --print-out-paths
/nix/store/dr9bj8qwflylq39zl5ajcx42cdrb1bv9-tt-metal-0.77.0
```

That path is the shared runtime this session used as the PRE-change runtime:
it is the binary that still contains the decode guard string (A3's
reproduction) and the string absence proof's negative side. The path is
unchanged from before the change, and the store path is input-addressed, so
the output is the same artifact the earlier receipts consumed.

Two source facts back the identity:

- `pkgs/tt-metal/default.nix` applies `./gdn-decode-fp32.patch` and no
  chunking patch; its last commits are `9ec34af` and `e1ac912`, both
  pre-change.
- `pkgs/tt-metal/gdn-decode-fp32.patch` still carries the guard
  (`num_heads exceeds the available compute grid`, 1 occurrence).
- The chunking lives only in the variant derivation
  `pkgs/tt-metal-gdn-grid/default.nix` plus `pkgs/tt-metal/gdn-decode-grid-chunk.patch`
  and `pkgs/tt-metal/gdn-prefill-scan-chunk.patch`, which the shared
  derivation does not reference.

The training-platform plan records still mount exactly
`/nix/store/dr9bj8qw…-tt-metal-0.77.0/.../p150_x2_mesh_graph_descriptor.textproto`,
so the platform consumes the same runtime after this change.

## Consumers

| Check | Result |
| --- | --- |
| `rwkv-layer-framework-parity` | build exit 0 |
| `rwkv-ttwkv7-decode-reader` | build exit 0 |
| `rwkv-ttwkv7-host-layout` | build exit 0 |
| `rwkv-ttwkv7-dispatch-abi` | build exit 0 |
| `rwkv-ttwkv7-observed-state-carry` | build exit 0 |
| `rwkv-ttwkv7-persistent-receipt-validator` | build exit 0 |
| `model-target-profile-factory` | build exit 0 |
| `model-target-consumer-generator` | build exit 0 |
| `model-target-candidate-report` | build exit 0 |
| `training-platform-nixos-example` | build exit 0 |
| `training-platform` | FAILS, pre-existing and independent |

## The `training-platform` failure is pre-existing

The failure is not caused by this change. The check derivation is byte-for-byte
the same before and after:

- at the pre-change commit `c095e3f` (with only the flake-evaluation repair
  this session already committed, since that commit cannot evaluate at all
  without it): `/nix/store/sl6im95x6fzx0k8z161zjlhnzqz8hc9l-training-platform-check.drv`;
- at this change's HEAD: `/nix/store/sl6im95x6fzx0k8z161zjlhnzqz8hc9l-training-platform-check.drv`.

Identical derivation means identical inputs, so the failure reproduces without
this change. Its log ends inside the check's own negative self-test section
(`validate-terminal-receipt` diagnostics) after
`training-platform-core self-test: PASS`; the failing step was not isolated in
this session and is recorded as an open pre-existing defect, not as a gate this
change broke.

## Non-claims

Device-free checks only. The physical RWKV-7 checks were not run: this change
does not touch their path, the shared runtime is unchanged, and the devices
were returned to the production service after the V8/V9 session.
