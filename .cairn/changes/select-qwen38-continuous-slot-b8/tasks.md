# Tasks: Select the admitted B=8 continuous-slot profile in the served configuration

## Phase 1: Attribution

- [x] [serial] A1 Confirm where the served configuration chooses the scheduler policy: the managed drop-in template, the profile service's package and profile validation, and the NixOS module's service command. r[tenstorrent.qwen38_p150x2.continuous_slot_served_profile]  (evidence/served-profile.md)
- [x] [serial] A2 Confirm the B=8 admission receipt covers the package and execution identity the deployment would serve, and that the historical owner command still names the B4 policy its package carries. r[tenstorrent.qwen38_p150x2.continuous_slot_served_profile]  (evidence/served-profile.md)

## Phase 2: Implementation

- [x] [serial] I1 Add a module option that selects the continuous-slot profile from the closed set {b4, b8} and refuses every other value before the service starts. r[tenstorrent.qwen38_p150x2.continuous_slot_served_profile]  (evidence/served-profile.md)
- [x] [serial] I2 Point the host-local managed drop-in at the admitted B=8 policy, keep the historical owner's command byte-exact on the B4 policy its package carries, and add a profile-and-drop-in switch operation to the profile service. r[tenstorrent.qwen38_p150x2.continuous_slot_served_profile]  (evidence/served-profile.md)
- [x] [serial] I3 Require the policy file a command names in the profile service's package and profile validation, so a profile that cannot serve its command is refused. r[tenstorrent.qwen38_p150x2.continuous_slot_served_profile]  (evidence/served-profile.md)
- [x] [parallel] I4 Extend the device-free checks: the module's B=8 command and the closed-set rejection of every other profile. r[tenstorrent.qwen38_p150x2.continuous_slot_served_profile]  (evidence/served-profile.md)

## Phase 3: Verification

- [x] [serial] V1 Run the device-free gates: the NixOS service check, the Qwen package check, and the repository quality gate. r[tenstorrent.qwen38_p150x2.continuous_slot_served_profile]  (evidence/served-profile.md)
- [x] [serial] V2 Verify the served profile on devices 0 and 1 at the served limits: the listener reports the B=8 profile, the admitted execution identity, eight live requests, and a slot refill. r[tenstorrent.qwen38_p150x2.continuous_slot_served_profile]  (evidence/served-profile.md)
- [ ] [serial] V3 Install the admitted package into the managed profile and make the systemd unit serve the B=8 policy. This step needs root on the host: the profile directory and the managed drop-in are root-owned, and this session's passwordless sudo covers only the unit's stop and start. The profile service's `switch-policy` operation performs it. r[tenstorrent.qwen38_p150x2.continuous_slot_served_profile]  (evidence/served-profile.md)

## Phase 4: Completion

- [x] [serial] D1 Record the served-profile evidence, the remaining host step, and every non-claim; move no served default until V3 runs. r[tenstorrent.qwen38_p150x2.continuous_slot_served_profile]  (evidence/served-profile.md)
