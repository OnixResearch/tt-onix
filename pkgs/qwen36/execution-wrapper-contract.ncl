# Immutable contract for the four served P150x2 execution entry points.
let NonEmptyStringC = std.contract.from_predicate (fun value =>
  std.is_string value && std.string.length value > 0
) in
let NonEmptyStringArrayC = std.contract.from_predicate (fun value =>
  std.is_array value
  && std.array.length value > 0
  && std.array.all (fun item => std.is_string item && std.string.length item > 0) value
) in
let ExactC = fun expected => std.contract.from_predicate (fun value => value == expected) in
{
  schema | ExactC 1 = 1,
  module | NonEmptyStringC = "qwen36_service.server",
  decode_mode | ExactC "candidate" = "candidate",
  mesh | ExactC "P150x2" = "P150x2",
  package_identity_kind | ExactC "nix-execution-artifact" = "nix-execution-artifact",
  entrypoints | NonEmptyStringArrayC = [
    "qwen36-p150x2-serve",
    "qwen36-p150x2-trace-candidate",
    "qwen36-p150x2-batch-admission",
    "qwen36-p150x2-trace-reuse-admission",
  ],
  required_profiles | NonEmptyStringArrayC = [
    "decode-profile.json",
    "batch-admission-profile.json",
    "trace-reuse-profile.json",
  ],
  required_environment | NonEmptyStringArrayC = [
    "HF_HUB_OFFLINE=1",
    "MESH_DEVICE=P150x2",
    "QWEN35_TEMP=0",
    "TT_METAL_HOME",
    "TT_METAL_RUNTIME_ROOT",
  ],
}
