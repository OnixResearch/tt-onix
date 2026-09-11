# Typed, fail-closed contract for the Qwen3.8 P150x2 TTFT control.
let required_root_count = 6 in
let minimum_retained_samples = 5 in
let PositiveInt =
  std.contract.from_predicate (fun value =>
    std.is_number value && value > 0 && std.number.is_integer value
  )
in
let NonNegativeInt =
  std.contract.from_predicate (fun value =>
    std.is_number value && value >= 0 && std.number.is_integer value
  )
in
let Positive =
  std.contract.from_predicate (fun value =>
    std.is_number value && value > 0
  )
in
let NonEmpty =
  std.contract.from_predicate (fun value =>
    std.is_string value && std.string.length value > 0
  )
in
let Revision =
  std.contract.from_predicate (fun value =>
    std.is_string value && std.string.is_match "^[0-9a-f]{40}$" value
  )
in
let Blake3 =
  std.contract.from_predicate (fun value =>
    std.is_string value && std.string.is_match "^[0-9a-f]{64}$" value
  )
in
let RetainedSamples =
  std.contract.from_predicate (fun value =>
    std.is_number value
    && std.number.is_integer value
    && value >= minimum_retained_samples
  )
in
let RouteSchedule =
  std.contract.from_predicate (fun value =>
    std.is_array value
    && std.array.length value == required_root_count
    && std.array.all std.is_string value
    && std.array.all (fun item => std.string.length item > 0) value
  )
in
let ProfileContract = {
  schema_version = 1,
  profile_id | NonEmpty,
  diagnostic_only = true,
  identities = {
    authority_blake3 | Blake3,
    checkpoint_revision | Revision,
    model_source_revision | Revision,
    tokenizer_revision | Revision,
  },
  topology = {
    expected_pcie_device_ids | Array NonNegativeInt,
    mesh_rows | PositiveInt,
    mesh_columns | PositiveInt,
    minimum_cpu | NonNegativeInt,
    maximum_cpu | NonNegativeInt,
    trace_region_size_bytes | PositiveInt,
    maximum_sequence_length | PositiveInt,
    kv_block_size | PositiveInt,
    request_cache_blocks | PositiveInt,
    trace_capacity_blocks | PositiveInt,
    page_table_blocks | PositiveInt,
  },
  workload = {
    prompt | NonEmpty,
    generated_tokens | PositiveInt,
  },
  roots = {
    route_schedule | RouteSchedule,
    warmup_samples | PositiveInt,
    retained_samples | RetainedSamples,
    maximum_within_root_spread_ratio | Positive,
    maximum_duplicate_root_ratio | Positive,
    material_regression_ratio | Positive,
    phase_sum_tolerance_seconds | Positive,
  },
}
in
{
  Profile = ProfileContract,
}
