# r[verify tenstorrent.training_platform.configuration.invalid]
# Negative fixture: an unsupported schema version.
let contracts = import "../contracts.ncl" in
let platform = import "../platform.ncl" in
platform
& {
  schema_version | force = 999,
} | contracts.PlatformConfigClosed
  | contracts.PlatformConfigDeepClosed
  | contracts.PlatformConfig
