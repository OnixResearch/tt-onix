# r[verify tenstorrent.training_platform.configuration.invalid]
# Negative fixture: a four-card mesh name for the two-card platform.
let contracts = import "../contracts.ncl" in
let platform = import "../platform.ncl" in
platform
& {
  mesh.name | force = "p150_x4",
} | contracts.PlatformConfigClosed | contracts.PlatformConfig
