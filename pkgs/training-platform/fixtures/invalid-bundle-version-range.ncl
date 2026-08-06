# r[verify tenstorrent.training_platform.runtime_bundle.invalid]
# Negative fixture: a version range instead of an exact version.
let contracts = import "../contracts.ncl" in
let manifest = import "../manifest.ncl" in
manifest
& {
  transformers.version | force = ">=4.0",
} | contracts.BundleClosed | contracts.BundleDeepClosed | contracts.Bundle
