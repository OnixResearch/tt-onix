# Negative fixture: parity needs a named tolerance for every required role.
# Removing adapter_delta must fail admission before compilation.
let contracts = import "../contracts.ncl" in
let base = import "base-profile.ncl" in
let stripped = std.record.remove "tolerances" base in
(stripped & { tolerances = std.record.remove "adapter_delta" base.tolerances })
  | contracts.ProfileClosed
  | contracts.Profile
