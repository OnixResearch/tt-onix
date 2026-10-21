## Why

The B=8 continuous-slot profile was admitted on 2026-10-21: its queue-inclusive
receipt passes every predeclared gate for a specific package and execution identity.
Admission was the precondition the sibling change set for the served configuration, and
that change deliberately moved no served default.

The deployment still serves the B=4 profile. The live owner command comes from the
managed drop-in that `scripts/qwen38-profile-service` writes, and it names
`continuous-slot-policy.json`. The package already exports the admitted B=8 policy
beside it, so the switch is a configuration change, not a new artifact.

## What Changes

- Add a `continuousSlotProfile` option to the Qwen NixOS module, closed to {b4, b8},
  that selects which packaged scheduler policy the service passes.
- Extend the device-free checks: the module's B=8 command and the closed-set
  rejection of every other profile.
- On the host, point the managed drop-in at the admitted B=8 policy, keep the
  historical owner's rollback command exact on the B4 policy its own package
  carries, and add a `switch-policy` operation to the profile service that moves
  the profile and the drop-in together and restores both on failure.
- Install the admitted package into the managed profile and verify the served
  profile on devices 0 and 1.

## Impact

- **Files**: `nixos/module.nix` and `flake-modules/onix-packages.nix` in the
  repository; the host-local `scripts/qwen38-profile-99.conf` and
  `scripts/qwen38-profile-service`, which the repository keeps out of version
  control as machine-specific operational artifacts; the adoption document, this
  change directory, and its evidence.
- **Testing**: the NixOS service check, the Qwen package check, the repository quality
  gate, and one physical served-profile verification on the two P150a devices.
- **Risk**: the served profile changes from four to eight live slots. Short requests
  stop waiting behind long ones, and long requests take about 20 % longer with pauses
  up to 5.5 s, exactly as the admission receipt measured.
