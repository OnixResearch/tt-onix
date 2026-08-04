# r[verify tenstorrent.training_platform.configuration.invalid]
# Negative fixture: an unknown top-level field.
let contracts = import "../contracts.ncl" in
let platform = import "../platform.ncl" in
platform
& {
  unexpected_field = "not admitted",
} | contracts.PlatformConfigClosed | contracts.PlatformConfig
