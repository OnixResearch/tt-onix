# Typed, fail-closed contracts for the continuous-slot admission comparison.
let PositiveIntC = fun maximum => std.contract.from_predicate (fun value =>
  std.is_number value
  && std.number.is_integer value
  && value > 0
  && value <= maximum
) in
let RatioC = std.contract.from_predicate (fun value =>
  std.is_number value && value > 0 && value <= 10
) in
let SecondsC = std.contract.from_predicate (fun value =>
  std.is_number value && value > 0 && value <= 3600
) in
let IdC = std.contract.from_predicate (fun value =>
  std.is_string value && std.string.is_match "^[a-z0-9][a-z0-9-]*$" value
) in
let TrueC = std.contract.from_predicate (fun value => value == true) in
let ClosedRecordC = fun allowed => std.contract.from_predicate (fun value =>
  std.is_record value
  && std.array.length (std.record.fields value) == std.array.length allowed
  && std.array.all (fun field => std.array.elem field allowed) (std.record.fields value)
) in
let DistinctClassesC = std.contract.from_predicate (fun value =>
  value.short_class != value.long_class
) in
{
  Closed = ClosedRecordC [
    "schema_version",
    "profile_id",
    "scheduler_profile_id",
    "workload",
    "gates",
  ],
  WorkloadClosed = ClosedRecordC [
    "workload_id",
    "repetitions",
    "short_class",
    "long_class",
  ],
  WorkloadCoherent = DistinctClassesC,
  GatesClosed = ClosedRecordC [
    "require_matched_identity",
    "require_all_completed",
    "require_exact_candidate_parity",
    "maximum_short_median_wall_ratio",
    "maximum_short_p95_wall_ratio",
    "maximum_long_p95_wall_ratio",
    "maximum_slowdown_ratio",
    "maximum_inter_token_gap_seconds",
    "maximum_median_inter_token_gap_ratio",
    "minimum_delivered_throughput_ratio",
    "minimum_startup_dram_free_ratio",
  ],
  Fields = {
    schema_version | std.contract.from_predicate (fun value => value == 1),
    profile_id | IdC,
    scheduler_profile_id | IdC,
    workload | {
      workload_id | IdC,
      repetitions | PositiveIntC 16,
      short_class | IdC,
      long_class | IdC,
    },
    gates | {
      require_matched_identity | TrueC,
      require_all_completed | TrueC,
      require_exact_candidate_parity | TrueC,
      maximum_short_median_wall_ratio | RatioC,
      maximum_short_p95_wall_ratio | RatioC,
      maximum_long_p95_wall_ratio | RatioC,
      maximum_slowdown_ratio | RatioC,
      maximum_inter_token_gap_seconds | SecondsC,
      maximum_median_inter_token_gap_ratio | RatioC,
      minimum_delivered_throughput_ratio | RatioC,
      minimum_startup_dram_free_ratio | RatioC,
    },
  },
}
