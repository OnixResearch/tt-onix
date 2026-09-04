let NonEmpty = std.contract.from_predicate (fun value =>
  std.is_string value && std.string.length value > 0
) in

{
  schema_version | Number = 1,
  required | Array NonEmpty = [
    "migration-boundary",
    "model-target-profile-factory",
    "rwkv-layer-framework-parity",
    "training-platform",
    "qwen36",
    "qwen36-authority-negative",
    "qwen36-nixos-service",
  ],
}
