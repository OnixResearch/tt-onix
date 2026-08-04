# r[impl tenstorrent.training_platform.configuration]
# r[impl tenstorrent.training_platform.runtime_bundle]
# Typed contracts for the dual-P150A TT-XLA training platform.
# The admission core in src/ re-checks every rule in Rust before launch.

let NonEmptyC = std.contract.from_predicate (fun value =>
  std.is_string value && std.string.length value > 0
) in

let PositiveC = std.contract.from_predicate (fun value =>
  std.is_number value && value > 0
) in

# Full-length lowercase hex content identity, for example BLAKE3 or sha256.
let HashHexC = std.contract.from_predicate (fun value =>
  std.is_string value && std.string.is_match "^[0-9a-f]{64}$" value
) in

let GitRevisionC = std.contract.from_predicate (fun value =>
  std.is_string value && std.string.is_match "^[0-9a-f]{40}$" value
) in

# OCI image digest. A tag alone is a mutable identity and fails this contract.
let ImageDigestC = std.contract.from_predicate (fun value =>
  std.is_string value && std.string.is_match "^sha256:[0-9a-f]{64}$" value
) in

# Exact package version. Range operators and mutable labels are rejected.
let ExactVersionC = std.contract.from_predicate (fun value =>
  std.is_string value
  && std.string.length value > 0
  && !(std.string.is_match "[><~*]" value)
  && value != "latest"
) in

let IsoDateC = std.contract.from_predicate (fun value =>
  std.is_string value && std.string.is_match "^[0-9]{4}-[0-9]{2}-[0-9]{2}" value
) in

let HttpsUrlC = std.contract.from_predicate (fun value =>
  std.is_string value && std.string.is_match "^https://" value
) in

let AbsolutePathC = std.contract.from_predicate (fun value =>
  std.is_string value && std.string.is_match "^/" value && !(std.string.is_match "\\.\\." value)
) in

let ImageC = {
  repository | NonEmptyC,
  digest | ImageDigestC,
  created | IsoDateC,
  base_os | NonEmptyC,
} in

let WheelC = {
  package | NonEmptyC,
  version | ExactVersionC,
  wheel_sha256 | HashHexC,
  python_abi | NonEmptyC,
  index | HttpsUrlC,
} in

let VersionedPackageC = {
  package | NonEmptyC,
  version | ExactVersionC,
  python_abi | NonEmptyC | optional,
} in

let HashedPackageC = {
  package | NonEmptyC,
  version | ExactVersionC,
  wheel_sha256 | HashHexC,
  python_abi | NonEmptyC,
} in

let SourceC = {
  repository | HttpsUrlC,
  revision | GitRevisionC,
  revision_date | IsoDateC,
} in

let DriverC = {
  name | NonEmptyC,
  version | ExactVersionC,
} in

let FirmwareC = {
  bundle_version | ExactVersionC,
  floor_version | ExactVersionC,
} in

let BundleC = {
  schema_version | PositiveC,
  bundle_id | NonEmptyC,
  recorded_date | IsoDateC,
  runtime_image | ImageC,
  pjrt_plugin | WheelC,
  tt_xla_source | SourceC,
  tt_blacksmith_source | SourceC,
  pytorch | HashedPackageC,
  pytorch_xla | HashedPackageC,
  jax | VersionedPackageC,
  jaxlib | VersionedPackageC,
  transformers | VersionedPackageC,
  driver | DriverC,
  firmware | FirmwareC,
} in

let DeviceClassC = std.contract.from_predicate (fun value =>
  value == "blackhole"
) in

let BoardClassC = std.contract.from_predicate (fun value =>
  value == "p150"
) in

let MeshNameC = std.contract.from_predicate (fun value =>
  value == "p150_x2"
) in

let DevicesC = {
  count | std.contract.from_predicate (fun value => value == 2),
  architecture | DeviceClassC,
  board_class | BoardClassC,
  device_root | AbsolutePathC,
} in

let MeshC = {
  name | MeshNameC,
  rows | std.contract.from_predicate (fun value => value == 1),
  columns | std.contract.from_predicate (fun value => value == 2),
  fabric_channels | PositiveC,
  descriptor_relative_path | NonEmptyC,
} in

let HugepagesC = {
  mount | AbsolutePathC,
  page_size | std.contract.from_predicate (fun value => value == "1G"),
  required_free_pages | PositiveC,
} in

let CacheC = {
  root | AbsolutePathC,
} in

let OutputC = {
  evidence_root | AbsolutePathC,
} in

# Closed-record contract: rejects any field outside the admitted set.
let ClosedRecordC = fun allowed =>
  std.contract.from_predicate (fun value =>
    std.is_record value
    && std.array.all (fun field => std.array.elem field allowed) (std.record.fields value)
  )
in

let bundle_fields = [
  "schema_version",
  "bundle_id",
  "recorded_date",
  "runtime_image",
  "pjrt_plugin",
  "tt_xla_source",
  "tt_blacksmith_source",
  "pytorch",
  "pytorch_xla",
  "jax",
  "jaxlib",
  "transformers",
  "driver",
  "firmware",
] in

let config_fields = [
  "schema_version",
  "bundle_id",
  "devices",
  "mesh",
  "hugepages",
  "cache",
  "output",
  "evidence",
  "smoke",
] in

let EvidenceExcludedC = std.contract.from_predicate (fun value =>
  std.is_array value
  && std.array.all
    (fun item =>
      std.array.elem
        item
        ["credentials", "raw_logs", "serial_numbers", "absolute_private_paths"])
    value
) in

let EvidenceC = {
  hash_algorithm | std.contract.from_predicate (fun value => value == "blake3"),
  exclude | EvidenceExcludedC,
} in

let PlatformConfigC = {
  schema_version | PositiveC,
  bundle_id | NonEmptyC,
  devices | DevicesC,
  mesh | MeshC,
  hugepages | HugepagesC,
  cache | CacheC,
  output | OutputC,
  evidence | EvidenceC,
  smoke | {
    timeout_seconds | PositiveC,
    collective_free | Bool,
  },
} in

{
  Bundle = BundleC,
  BundleClosed = ClosedRecordC bundle_fields,
  PlatformConfig = PlatformConfigC,
  PlatformConfigClosed = ClosedRecordC config_fields,
  HashHex = HashHexC,
  GitRevision = GitRevisionC,
  ImageDigest = ImageDigestC,
  ExactVersion = ExactVersionC,
  AbsolutePath = AbsolutePathC,
}
