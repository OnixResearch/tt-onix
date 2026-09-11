# r[impl tenstorrent.training_platform.configuration]
# One closed policy projection for package-owned Rust admission.
let contracts = import "contracts.ncl" in
{
  schema_version = 1,
  manifest = import "manifest.ncl",
  configuration = import "platform.ncl",
} | contracts.AdmissionPolicyClosed | contracts.AdmissionPolicy
