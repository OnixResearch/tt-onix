let NonEmpty = std.contract.from_predicate (fun value =>
  std.is_string value && std.string.length value > 0
) in

let PositiveInt = std.contract.from_predicate (fun value =>
  std.is_number value && value > 0
) in

let Positive = std.contract.from_predicate (fun value =>
  std.is_number value && value > 0
) in

let NonNegativeInt = std.contract.from_predicate (fun value =>
  std.is_number value && value >= 0
) in

let UnitFraction = std.contract.from_predicate (fun value =>
  std.is_number value && value > 0 && value <= 1
) in

let Revision = std.contract.from_predicate (fun value =>
  std.is_string value && std.string.is_match "^[0-9a-f]{40}$" value
) in

let Blake3 = std.contract.from_predicate (fun value =>
  std.is_string value && std.string.is_match "^[0-9a-f]{64}$" value
) in

let ExactlyTwoDevices = std.contract.from_predicate (fun value =>
  std.is_array value && std.array.length value == 2
) in

let ExactlyFour = std.contract.from_predicate (fun value =>
  std.is_array value && std.array.length value == 4
) in

let ProfileContract = {
  schema_version | PositiveInt,
  profile_id | NonEmpty,
  diagnostic_only = true,
  identities = {
    authority_blake3 | Blake3,
    checkpoint_revision | Revision,
    model_source_revision | Revision,
    tokenizer_revision | Revision,
  },
  topology = {
    expected_pcie_device_ids | Array NonNegativeInt | ExactlyTwoDevices,
    mesh_rows | PositiveInt,
    mesh_columns | PositiveInt,
  },
  workload = {
    modes | Array NonEmpty | ExactlyFour,
    execution_order | Array NonEmpty | ExactlyFour,
    samples_per_root | PositiveInt,
    warmup_runs | PositiveInt,
  },
  gates = {
    require_exact_token_parity = true,
    require_exact_text_parity = true,
    require_clean_reset = true,
    require_distinct_state_slices = true,
    maximum_duplicate_spread_ratio | Positive,
    maximum_gdn_median_ratio | UnitFraction,
    maximum_full_step_median_ratio | UnitFraction,
  },
}
in

{
  Profile = ProfileContract,
}
