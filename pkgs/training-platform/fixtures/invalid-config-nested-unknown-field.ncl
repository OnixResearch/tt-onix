# r[verify tenstorrent.training_platform.configuration.invalid]
# Negative fixture: an unknown nested device field.
let contracts = import "../contracts.ncl" in
let platform = import "../platform.ncl" in
platform
& {
  devices.unreviewed_device_override = true,
} | contracts.PlatformConfigClosed
  | contracts.PlatformConfigDeepClosed
  | contracts.PlatformConfig
