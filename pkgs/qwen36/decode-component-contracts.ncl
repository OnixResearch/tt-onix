let NonEmpty =
  std.contract.from_predicate (fun value =>
    std.is_string value && std.string.length value > 0
  )
in

let PositiveInt =
  std.contract.from_predicate (fun value =>
    std.is_number value && value > 0
  )
in

let NonNegativeInt =
  std.contract.from_predicate (fun value =>
    std.is_number value && value >= 0
  )
in

let Positive =
  std.contract.from_predicate (fun value =>
    std.is_number value && value > 0
  )
in

let UnitFraction =
  std.contract.from_predicate (fun value =>
    std.is_number value && value > 0 && value <= 1
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

let ExactlyThree =
  std.contract.from_predicate (fun value =>
    std.is_array value && std.array.length value == 3
  )
in

let ProfileContract = {
  profile_id | NonEmpty,
  diagnostic_only = true,
  execution_path | NonEmpty,
  samples_per_group | PositiveInt,
  warmup_replays | PositiveInt,
  maximum_spread_ratio | Positive,
  trace_capture_timeout_seconds | PositiveInt,
  closure = {
    minimum_ratio | UnitFraction,
    maximum_ratio | Positive,
  },
  selection = {
    minimum_weighted_share | UnitFraction,
    minimum_dominance_ratio | Positive,
  },
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
  },
  architecture = {
    layer_count | PositiveInt,
    gdn_layer_count | PositiveInt,
    full_attention_layer_count | PositiveInt,
    mlp_layer_count | PositiveInt,
  },
  representatives = {
    gdn | Array NonNegativeInt | ExactlyThree,
    full_attention | Array NonNegativeInt | ExactlyThree,
    mlp | Array NonNegativeInt | ExactlyThree,
  },
  prompt | NonEmpty,
}
in

{
  Profile = ProfileContract,
}
