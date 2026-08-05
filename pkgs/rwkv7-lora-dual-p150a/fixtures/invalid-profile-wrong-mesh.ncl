# Negative fixture: the logical mesh must be the 1x2 dual-replica mesh.
let contracts = import "../contracts.ncl" in
let base = import "base-profile.ncl" in
(std.record.remove "device" base) & {
  device = (std.record.remove "logical_mesh" base.device) & {
    logical_mesh = { rows = 1, columns = 1 },
  },
}
| contracts.ProfileClosed | contracts.Profile
