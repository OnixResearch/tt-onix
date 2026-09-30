# Typed, fail-closed contracts for the hybrid prefix-cache admission comparison.
let RatioC = std.contract.from_predicate (fun value => std.is_number value && value > 0 && value <= 10) in
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
let RootMinimumC = std.contract.from_predicate (fun value =>
  std.is_number value && std.number.is_integer value && value >= 2048 && value % 1024 == 0
) in
let ClosedRecordC = fun allowed => std.contract.from_predicate (fun value =>
  std.is_record value
  && std.array.length (std.record.fields value) == std.array.length allowed
  && std.array.all (fun field => std.array.elem field allowed) (std.record.fields value)
) in
# Cold and cached roots alternate in ABBA order, so linear load drift cancels.
let AbbaOrderC = std.contract.from_predicate (fun value => value == ["cold", "cached", "cached", "cold"]) in
{
  Closed = ClosedRecordC ["schema_version", "profile_id", "prefix_cache_profile_id", "root_order", "workloads", "gates"],
  SharedClosed = ClosedRecordC [
    "schema_version", "profile_id", "prefix_cache_profile_id", "root_order",
    "workloads", "gates", "shared_root_minimum_tokens", "state_slots",
  ],
  ToolClosed = ClosedRecordC [
    "schema_version", "profile_id", "prefix_cache_profile_id", "root_order",
    "workloads", "gates", "shared_root_minimum_tokens", "state_slots",
    "workload_id", "require_changed_tool_miss",
  ],
  GatesClosed = ClosedRecordC [
    "require_matched_identity",
    "require_exact_parity",
    "require_resumed_hits",
    "maximum_hit_median_ttft_ratio",
    "maximum_hit_median_wall_ratio",
    "maximum_miss_median_wall_ratio",
    "maximum_miss_wall_ratio",
    "maximum_startup_dram_drop_bytes",
  ],
  Fields = {
    schema_version | std.contract.from_predicate (fun value => value == 2),
    profile_id | IdC,
    prefix_cache_profile_id | IdC,
    root_order | AbbaOrderC,
    workloads | Array { conversations | CountC, turns | CountC },
    gates | {
      require_matched_identity | TrueC,
      require_exact_parity | TrueC,
      require_resumed_hits | TrueC,
      maximum_hit_median_ttft_ratio | RatioC,
      maximum_hit_median_wall_ratio | RatioC,
      maximum_miss_median_wall_ratio | RatioC,
      maximum_miss_wall_ratio | RatioC,
      maximum_startup_dram_drop_bytes | BytesC,
    },
  },
  SharedFields = {
    schema_version | std.contract.from_predicate (fun value => value == 3),
    profile_id | IdC,
    prefix_cache_profile_id | IdC,
    root_order | AbbaOrderC,
    workloads | Array { conversations | CountC, turns | CountC },
    shared_root_minimum_tokens | RootMinimumC,
    state_slots | CountC,
    gates | {
      require_matched_identity | TrueC,
      require_exact_parity | TrueC,
      require_resumed_hits | TrueC,
      maximum_hit_median_ttft_ratio | RatioC,
      maximum_hit_median_wall_ratio | RatioC,
      maximum_miss_median_wall_ratio | RatioC,
      maximum_miss_wall_ratio | RatioC,
      maximum_startup_dram_drop_bytes | BytesC,
    },
  },
  ToolFields = {
    schema_version | std.contract.from_predicate (fun value => value == 4),
    profile_id | IdC,
    prefix_cache_profile_id | IdC,
    root_order | AbbaOrderC,
    workloads | Array { conversations | CountC, turns | CountC },
    shared_root_minimum_tokens | RootMinimumC,
    state_slots | CountC,
    workload_id | IdC,
    require_changed_tool_miss | TrueC,
    gates | {
      require_matched_identity | TrueC,
      require_exact_parity | TrueC,
      require_resumed_hits | TrueC,
      maximum_hit_median_ttft_ratio | RatioC,
      maximum_hit_median_wall_ratio | RatioC,
      maximum_miss_median_wall_ratio | RatioC,
      maximum_miss_wall_ratio | RatioC,
      maximum_startup_dram_drop_bytes | BytesC,
    },
  },
}
