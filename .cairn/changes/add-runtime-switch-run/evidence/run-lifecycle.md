# Run lifecycle evidence

Captured 2026-10-26 on britton-desktop, unprivileged, with a fake owner helper, a
fixture job package, and temporary state roots under `/var/tmp`. No unit, device,
or real `tt-smi` was touched.

## Command

```text
TT_MODEL_REGISTRY=<fixture registry> \
TT_MODEL_STATE_ROOT=/var/tmp/rt/runstate \
TT_MODEL_GC_ROOTS=/var/tmp/rt/runroots \
TT_MODEL_OWNER=/var/tmp/rt/fake-owner-ok \
MESH_DEVICE=leaked FAKE_JOB_ENV=caller \
  /nix/store/<tt-model>/bin/tt-model run fake-job \
    --package /nix/store/…-fixture-job \
    --repo-root /nix/store/…-fixture-runtime-registry-repo \
    --json -- --receipt /var/tmp/rt/job-receipt.json
```

Exit 0. The job's own output went to the runner's stderr and proves the
environment contract: the profile's `FAKE_JOB_ENV=1` reached the job, and the
caller's `MESH_DEVICE=leaked` did not.

```text
job env=1 mesh=unset
```

## Event written

```json
{
  "event": {
    "argv": [
      "/nix/store/…-fixture-job/bin/job-run",
      "--receipt",
      "/var/tmp/rt/job-receipt.json"
    ],
    "argv_blake3": "75ab934a75d9181b6b2b0dee8cee580d1d03fc2f0a7cc4c0b4f5a6425bfeee2b",
    "exit_status": 0,
    "finished": "2026-10-26T15:40:20.871016+00:00",
    "owner_stopped": true,
    "package": "/nix/store/…-fixture-job",
    "profile": "fake-job",
    "receipt": {
      "blake3": "61b9def581c87b2e4f5406ba9570c490bd2da2afd8d39c22e3be730d2e538663",
      "path": "/var/tmp/rt/job-receipt.json"
    },
    "reset_performed": true,
    "schema": "onix.tt-model.run.v1",
    "started": "2026-10-26T15:40:20.867098+00:00"
  },
  "event_path": "/var/tmp/rt/runstate/runs/20261026T154020.867098Z-fake-job.json",
  "last_run": "/var/tmp/rt/runstate/last-run.json"
}
```

`owner_stopped` is true because the fake helper reported a live owner PID;
`reset_performed` is true because the profile is TT-Metal (`on_release =
required-before-native`). `active.json` does not exist in that state root.

## Status reports the run, not an owner

```text
verdict: degraded
active record: absent (/var/tmp/rt/runstate/active.json)
last run: fake-job /nix/store/…-fixture-job exit 0 (2026-10-26T15:40:20.867098+00:00)
operator symlink: absent
nix profile: absent
unit: unavailable (unit probe disabled by TT_MODEL_SYSTEMCTL)
```

The verdict is `degraded` only because the unit probe is disabled in this run; the
run event changes no verdict, and the active record stays absent.

## Sandbox refusals (package install check)

Each is asserted by its message, with temporary roots and fake helpers:

| Case | Message |
|---|---|
| serve profile | `qwen38 is a serve profile; use \`tt-model switch\`` |
| structural failure, even with `--force` | `run refused: check:` |
| missing helper | `the owner helper is unavailable` |
| foreign holder | `run refused: card-holder` |
| helper `cards` fails | `cards failed with status 3` |
| admission failure without `--force` | `run refused: check:` |
| admission failure with `--force` and `TT_MODEL_UNATTENDED=1` | `--force is refused when TT_MODEL_UNATTENDED=1` |
| held run lock | `another run operation holds the lock` |

The sandbox also runs a forced admission bypass to completion, asserts the second
run event, and asserts `active.json` is absent from the run state root.

## Non-claims

The helper is fake, the job is a fixture, and no card was opened. The live
bounded run against a real owner and real boards is deferred to a separately
authorized step.
