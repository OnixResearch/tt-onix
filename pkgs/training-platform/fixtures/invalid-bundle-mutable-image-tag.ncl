# r[verify tenstorrent.training_platform.runtime_bundle.invalid]
# Negative fixture: a mutable image tag instead of an immutable digest.
let contracts = import "../contracts.ncl" in
let manifest = import "../manifest.ncl" in
manifest
& {
  runtime_image.digest | force = "latest",
} | contracts.BundleClosed | contracts.Bundle
