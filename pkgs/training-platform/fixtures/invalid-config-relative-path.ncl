# r[verify tenstorrent.training_platform.configuration.invalid]
# Negative fixture: a relative cache path.
let contracts = import "../contracts.ncl" in
let platform = import "../platform.ncl" in
platform
& {
  cache.root | force = "var/lib/tenstorrent-training/cache",
} | contracts.PlatformConfigClosed | contracts.PlatformConfigDeepClosed | contracts.PlatformConfig
