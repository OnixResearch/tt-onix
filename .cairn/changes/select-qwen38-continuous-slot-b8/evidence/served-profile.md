# The served B=8 profile: verification on devices 0 and 1 (2026-10-21)

The B=8 continuous-slot profile is admitted (receipt
`pkgs/qwen36/evidence/2026-10-21/qwen38-continuous-slot-admission-b8-receipt.json`,
BLAKE3 `2689347235ed11105d39d284cd5f0a3b37e0809ae101bc22c31354f97edc2280`). This
file records the served-profile verification and the one host step that still needs
root.

## What was verified

The admitted package was served with the admitted B=8 policy at the production
limits, and eight concurrent requests were sent to it.

- Package: `/nix/store/inkfvsjqchqp7bki76asdib3b57nw6kj-qwen38-0.1.0`.
- Execution identity reported by the service:
  `/nix/store/fk8p9n0xdx62yx8hr1jnccgjfyp2j8s0-qwen36-execution-identity-0.1.0` — the
  identity the admission receipt admitted.
- Runtime: `/nix/store/7hplkynh047dsk98cwqpfsaaq6lwyy0m-tt-metal-0.77.0`.
- Scheduler policy: `share/qwen36/continuous-slot-policy-b8.json`.
- Prefix cache: enabled, as the served unit has it.
- Limits: `--max-sequence-length 131072 --max-generation-tokens 8192`, the served
  unit's values.
- Devices: physical PCIe 0 and 1, `simulated: false` in every response.

| Observation | Value |
| --- | --- |
| Requests sent concurrently | 8 |
| Failed requests | 0 |
| Scheduler profile in the response evidence | `qwen38_p150x2-continuous-slot-b8-v1` |
| Distinct slot indices | 0, 1, 2, 3, 4, 5 (one slot refilled after its request finished) |
| Client-observed walls | 0.774 s to 2.731 s |
| Answers | Paris, Red, 4, Iron, 7, Mars, Two hydrogens one oxygen, Summer |

The service reported the admitted identity, the B=8 profile, eight live requests with
no queueing behind a batch, and one slot refill inside the same run.

## Startup at the served limits

- Readiness: 406 s; the startup warmup reported 346.9 s over widths 1 and 4, and the
  masked prefill traces were captured for buckets 128, 256, 512, and 1024.
- Device DRAM after startup: 9,501,091,840 B free of 23,603,897,344 B allocated. The
  served configuration keeps the prefix cache, whose saved state slots account for the
  difference from the admission measurement's 9,530,484,736 B.

## The managed profile switch, performed

The profile and the systemd unit are root-owned on this host, so the move was made by
the owner with the profile service's `switch-policy` operation:

```text
sudo scripts/qwen38-profile-service switch-policy \
  /nix/store/inkfvsjqchqp7bki76asdib3b57nw6kj-qwen38-0.1.0
```

The first attempt failed before any change: the owner-control helper isolates an
*active* owner only, and the unit was stopped. The operation was corrected to start the
current owner when it is not active, and the second attempt completed:

- The named profile now holds `/nix/store/inkfvsjqchqp7bki76asdib3b57nw6kj-qwen38-0.1.0`.
- The managed drop-in names
  `/nix/var/nix/profiles/qwen38-p150x2/share/qwen36/continuous-slot-policy-b8.json`.
- The unit restarted with `NRestarts = 0` and reported ready after a 475.3 s startup
  warmup over widths 1 and 4.

The live unit on `127.0.0.1:8000` was then verified with the same eight concurrent
requests:

| Observation | Value |
| --- | --- |
| Requests sent concurrently | 8 |
| Failed requests | 0 |
| Scheduler profile | `qwen38_p150x2-continuous-slot-b8-v1` |
| Decode path | `continuous-slot` |
| Distinct slot indices | 0, 1, 2, 3, 4, 5 (one slot refilled inside the run) |
| Execution identity | `/nix/store/fk8p9n0xdx62yx8hr1jnccgjfyp2j8s0-qwen36-execution-identity-0.1.0` |
| Runtime identity | `/nix/store/7hplkynh047dsk98cwqpfsaaq6lwyy0m-tt-metal-0.77.0` |
| Devices | physical PCIe 0 and 1, `simulated: false` |
| Client-observed walls | 1.274 s to 2.921 s |
| Answers | Paris, Red, 4, Iron, 7, Mars, Two hydrogens one oxygen, Summer |

The unit was then stopped, as agreed: it is `inactive` with no process holding either
device node, and the owner-control capability validation passes.

## Non-claims

- The verification is one bounded concurrent workload at the served limits, twice:
  once against a direct run of the admitted package and once against the live unit.
  It is not a throughput, latency, or stability claim.
- The unit is stopped after the verification. The switch is in place; the service
  starts on the B=8 profile when it is started.
- The admission receipt is the authority for the profile's bounds; this file only
  records that the admitted identity serves that profile as configured.
