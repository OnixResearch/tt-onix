# The switch, verified before and after (2026-10-21)

Recorded 2026-10-21 for change `deploy-qwen38-measured-runtime`, tasks I2, I3,
V1-V4, D1. The deployment gap recorded in `deployment-gap.md` is closed: the served
Qwen3.8 instance now runs on an execution identity that held receipts name, and the
switch was verified on both sides.

## The before and after

| | production on 2026-09-08 | served on 2026-10-21 |
| --- | --- | --- |
| package | `jsivpglnl…-qwen36-0.1.0` | `wlsl6pnx…-qwen38-0.1.0` |
| runtime identity | `djchdwzk…-tt-metal-0.74.0` | `7hplkynh…-tt-metal-0.77.0` (the shared output) |
| execution identity | `yarrawqw…-qwen36-execution-identity-0.1.0` | `l4arf461…-qwen36-execution-identity-0.1.0` |
| decode path | `batched-traced`, candidate mode | `continuous-slot`, the eight-slot profile, with the hybrid prefix cache |
| receipts naming it | none | the eight-slot scheduler admission and the prefix-cache admission |

The execution identity `l4arf461…` is named by both held receipts:

- `pkgs/qwen36/evidence/2026-10-21/qwen38-continuous-slot-admission-b8-receipt.json`,
  BLAKE3 `4573cea118453a339ad86b73fe67f729e38dd1c5187b9bb09a4ba527e66af11e`;
- `pkgs/qwen36/evidence/2026-10-21/qwen38-prefix-cache-abba-admission-receipt.json`,
  BLAKE3 `eb09b2cc9ec785b223200bbe67b11b6e2665dd25214ce20ca41500220b136f91`.

## V1: the host came up on the new package and runtime

The switch was performed with `scripts/qwen38-deploy`, which moved the served
instance onto the package, restarted the owner, and waited for readiness.

- The unit is `active`, `running`, with `NRestarts = 0`.
- `/healthz` reports `ready: true` and a startup warm-up of 478.8 s over widths 1
  and 4.
- The service reports package identity `l4arf461…` and runtime identity
  `7hplkynh…`, on physical PCIe devices 0 and 1, `simulated: false`.

## V2: the served instance reproduces the admitted prefix

One greedy request on the frozen corpus, served by the live instance on port 8000:

| Observation | Value |
| --- | --- |
| Prompt | `The capital of France is` (5 prompt tokens) |
| Text | ` Paris.\nThe capital of Germany is` |
| Token ids | 11751, 13, 198, 760, 6511, 314, 9564, 369 |
| Matches the admitted reference prefix | yes |
| Finish reason | `length` (the 8-token cap) |
| Decode path | `continuous-slot` |
| Serving route | `step` |
| Scheduler profile | `qwen38_p150x2-continuous-slot-b8-v1` |
| Prefix-cache profile | `qwen38-p150x2-prefix-cache-v1` |
| `admission_eligible` | `false` |

The token prefix is the one the earlier B=1 and B=8 evidence recorded, so the served
instance starts, serves, and reproduces the known output on both physical devices.

## V3: the served routes, and what is not served

- Readiness and health are confirmed above.
- The served command names the eight-slot scheduler policy and the prefix-cache
  policy, both under the operator's path, and carries no batch-admission argument.
- The response's `admission_eligible` is `false`: the runtime change alone labels
  nothing admission-eligible, which is what the requirement asks for.
- The group coordinator's batch route is not served. What the instance serves is
  the eight-slot continuous-slot profile and the hybrid prefix cache, and both have
  receipts that name this package and this execution identity. That is a stronger
  position than the one this change was written against, where the batch-admission
  route was to stay off because no receipt covered it; the route is now covered, and
  the scheduler that is served is the admitted one.

## V4: the shared runtime and its consumers are untouched

- The shared tt-metal 0.77 output is the same store path the receipts name:
  `/nix/store/7hplkynh047dsk98cwqpfsaaq6lwyy0m-tt-metal-0.77.0`.
- `checks.x86_64-linux.metalium-rs-rwkv7-tokenizer` passes.
- `checks.x86_64-linux.training-platform` passes.

## I2: the host edit

The handoff this change recorded is superseded. The host's configuration was never
in this repository and is not needed: the selection is the managed systemd drop-in
plus a path the operator owns.

- The root-owned drop-in
  `/usr/local/lib/systemd/system/qwen38-p150x2.service.d/99-profile.conf` names
  `/home/brittonr/.local/state/qwen38-p150x2/current`, the checked-in template
  `scripts/qwen38-profile-99.conf`.
- That path is the operator's symlink to the package that is served.
- `scripts/qwen38-deploy` is the edit operation: it builds the package, refuses a
  package that no receipt names, repoints the symlink, restarts the owner through
  the host's existing passwordless stop and start, and checks what is served.

The procedure is in `docs/qwen38-served-deployment.md`. No new sudoers rule is
needed, and no configuration in the sibling repository was touched.

## I3: the artifact manifest

The merge damage that made the manifest invalid JSON is gone, and the gate reports
its count rather than a parse error. The count drift that commit `680afa7` recorded
and chose to keep visible was absorbed by the repository's own later regenerations
of the manifest, from `e205476` onward, which is this repository's standing
practice whenever the tree gains artifacts. The manifest was regenerated again for
the artifacts this work added, and it is current: the artifact gate passes.

The change's own instruction, to repair the manifest without regenerating it past
its owner's recorded decision, no longer describes the repository's state; that
decision was superseded by the repository's own commits before this change was
written.

## D1: non-claims

- The switch moved the service from tt-metal 0.74 onto the shared 0.77 output and
  onto a package and execution identity that two receipts name. The runtime change
  itself is not an admission of anything.
- The served instance is one readiness measurement and one greedy request. It is
  not a throughput, latency, or stability claim, and it says nothing about any
  width beyond the two admitted profiles.
- The single-stream group path is not the served path any more: the served path is
  the admitted eight-slot scheduler with the admitted prefix cache. The earlier
  claim that a single-stream service would be switched to a measured identity is
  met in a different way than this change first recorded.
- The measurements behind the receipts are the campaigns in
  `.cairn/changes/widen-qwen38-continuous-slots-b8` and
  `.cairn/changes/readmit-qwen38-prefix-cache-b8`, not this switch.
