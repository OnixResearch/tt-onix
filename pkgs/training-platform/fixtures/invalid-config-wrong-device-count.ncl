# r[verify tenstorrent.training_platform.configuration.invalid]
# Negative fixture: one device instead of the required two.
let contracts = import "../contracts.ncl" in
let platform = import "../platform.ncl" in
platform
& {
  devices.count | force = 1,
} | contracts.PlatformConfigClosed | contracts.PlatformConfig
