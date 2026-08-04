# Negative fixture: a full-checkpoint artifact format conflicts with the
# CPU adapter contract. Admission must fail with a contract violation
# before compilation.
let contracts = import "../contracts.ncl" in
let base = import "base-profile.ncl" in
let stripped = std.record.remove "artifact_policy" base in
(
  stripped
  & {
    artifact_policy = {
      format = "full-checkpoint",
      artifact_root = "/var/lib/tenstorrent-training/artifacts/rwkv7-lora-p150a",
    },
  }
) | contracts.ProfileClosed
  | contracts.Profile
