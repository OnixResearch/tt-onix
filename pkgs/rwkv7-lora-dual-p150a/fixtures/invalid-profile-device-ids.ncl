# Negative fixture: the runner is fail-closed on physical device IDs 0 and 1.
let contracts = import "../contracts.ncl" in
let base = import "base-profile.ncl" in
(std.record.remove "device" base) & {
  device = (std.record.remove "physical_device_ids" base.device) & {
    physical_device_ids = [0, 2],
  },
}
| contracts.ProfileClosed | contracts.Profile
