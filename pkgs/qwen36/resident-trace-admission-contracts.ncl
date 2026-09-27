# Typed, fail-closed contracts for the resident slot-trace admission comparison.
let RatioC = std.contract.from_predicate (fun value => std.is_number value && value > 0 && value <= 10) in
let ShareC = std.contract.from_predicate (fun value => std.is_number value && value >= 0 && value < 1) in
let IdC = std.contract.from_predicate (fun value =>
  std.is_string value && std.string.is_match "^[a-z0-9][a-z0-9-]*$" value
) in
let TrueC = std.contract.from_predicate (fun value => value == true) in
let CountC = std.contract.from_predicate (fun value =>
  std.is_number value && std.number.is_integer value && value >= 1 && value <= 16
) in
let BytesC = std.contract.from_predicate (fun value =>
  std.is_number value && std.number.is_integer value && value >= 0
) in
let ClosedRecordC = fun allowed => std.contract.from_predicate (fun value =>
  std.is_record value
  && std.array.length (std.record.fields value) == std.array.length allowed
  && std.array.all (fun field => std.array.elem field allowed) (std.record.fields value)
) in
# Baseline and candidate roots alternate in ABBA order, so linear load drift cancels.
let AbbaOrderC = std.contract.from_predicate (fun value =>
  value == ["baseline", "candidate", "candidate", "baseline"]
) in
{
  Closed = ClosedRecordC ["schema_version", "profile_id", "prefix_cache_profile_id", "root_order", "workloads", "gates"],
  WorkloadClosed = ClosedRecordC ["workload_id", "conversations", "turns"],
  GatesClosed = ClosedRecordC [
    "require_matched_identity",
    "require_exact_parity",
    "require_matching_prefix_cache",
    "maximum_candidate_capture_share",
    "maximum_median_wall_ratio",
    "maximum_wall_ratio",
    "maximum_median_ttft_ratio",
    "maximum_startup_dram_drop_bytes",
  ],
  Fields = {
    schema_version | std.contract.from_predicate (fun value => value == 1),
    profile_id | IdC,
    prefix_cache_profile_id | IdC,
    root_order | AbbaOrderC,
    workloads | Array { workload_id | IdC, conversations | CountC, turns | CountC },
    gates | {
      require_matched_identity | TrueC,
      require_exact_parity | TrueC,
      require_matching_prefix_cache | TrueC,
      maximum_candidate_capture_share | ShareC,
      maximum_median_wall_ratio | RatioC,
      maximum_wall_ratio | RatioC,
      maximum_median_ttft_ratio | RatioC,
      maximum_startup_dram_drop_bytes | BytesC,
    },
  },
}
