let NonEmpty =
  std.contract.from_predicate (fun value =>
    std.is_string value && std.string.length value > 0
  )
in

let Positive =
  std.contract.from_predicate (fun value =>
    std.is_number value && value > 0
  )
in

let NonNegative =
  std.contract.from_predicate (fun value =>
    std.is_number value && value >= 0
  )
in

let RevisionContract =
  std.contract.from_predicate (fun value =>
    std.is_string value && std.string.is_match "^[0-9a-f]{40}$" value
  )
in

let Blake3Contract =
  std.contract.from_predicate (fun value =>
    std.is_string value && std.string.is_match "^[0-9a-f]{64}$" value
  )
in

let NixHashContract =
  std.contract.from_predicate (fun value =>
    std.is_string value && std.string.is_match "^sha256-[A-Za-z0-9+/]{43}=$" value
  )
in

let Url =
  std.contract.from_predicate (fun value =>
    std.is_string value
    && (std.string.is_match "^https://" value || std.string.is_match "^fixture://" value)
  )
in

let StateKindContract =
  std.contract.from_predicate (fun value =>
    value == "recurrent_matrix"
    || value == "kv_cache"
    || value == "hybrid_kv_recurrent"
  )
in

let StateComponentKindContract =
  std.contract.from_predicate (fun value =>
    value == "recurrent_matrix" || value == "kv_cache"
  )
in

let PrecisionContract =
  std.contract.from_predicate (fun value =>
    value == "bf16_checkpoint_fp32_reference"
    || value == "bf16_fixture"
    || value == "bf4_experts_bf8_attention_bf16_activations_candidate"
  )
in

let PositiveTolerance =
  std.contract.from_predicate (fun value =>
    std.is_number value && value > 0
  )
in

let ProfileArtifact = {
  role | NonEmpty,
  name | NonEmpty,
  url | Url,
  nix_hash | NixHashContract,
  blake3 | Blake3Contract,
}
in

# r[impl tenstorrent.model_targets.hybrid_profile_contract]
let StateComponent = {
  kind | StateComponentKindContract,
  layer_count | Positive,
  per_layer_element_count | Positive,
  carry_semantics | NonEmpty,
}
in

let ModelProfileContract = {
  target_name | NonEmpty,
  runnable | Bool,
  model
    | {
      id | NonEmpty,
      revision | RevisionContract,
      license | NonEmpty,
      architecture_family | NonEmpty,
      checkpoint_layout | NonEmpty,
      checkpoint_shard_count | optional | Positive,
      precision | PrecisionContract,
    },
  architecture
    | optional
    | {
      linear_attention_layer_count | NonNegative,
      full_attention_layer_count | NonNegative,
      other_layer_count | NonNegative,
      routed_expert_count | NonNegative,
      routed_experts_per_token | NonNegative,
      shared_expert_count | NonNegative,
      full_attention_query_head_count | NonNegative,
      full_attention_kv_head_count | NonNegative,
      full_attention_head_size | NonNegative,
      linear_key_head_count | NonNegative,
      linear_value_head_count | NonNegative,
      linear_head_size | NonNegative,
      causal_conv_kernel_size | NonNegative,
    },
  state
    | {
      kind | StateKindContract,
      per_layer_element_count | Positive,
      carry_semantics | NonEmpty,
      components | optional | Array StateComponent,
    },
  resources
    | {
      checkpoint_bytes | Positive,
      layer_count | Positive,
      hidden_size | Positive,
      vocabulary_size | Positive,
      maximum_prompt_tokens | Positive,
    },
  adapters
    | {
      architecture | NonEmpty,
      evidence | NonEmpty,
    },
  topology
    | {
      id | NonEmpty,
      mesh_shape | Array Positive,
      physical_device_ids | Array NonNegative,
      replication | NonEmpty,
    },
  consumer
    | {
      dimensions
        | {
          head_count | Positive,
          head_size | Positive,
          intermediate_size | Positive,
          complete_state_element_count | Positive,
        },
      precisions
        | {
          cpu_reference | NonEmpty,
          runtime_arithmetic | NonEmpty,
          ttwkv7_boundary | NonEmpty,
        },
      tolerances
        | {
          harness_oracle | PositiveTolerance,
          harness_replay | PositiveTolerance,
          state_carry_divergence_floor | PositiveTolerance,
          final_hidden | PositiveTolerance,
          logits | PositiveTolerance,
          recurrent_state | PositiveTolerance,
          control_divergence_floor | PositiveTolerance,
          lora_gradient_agreement | PositiveTolerance,
          lora_finite_difference | PositiveTolerance,
        },
      adapter_format | NonEmpty,
    },
  artifacts | Array ProfileArtifact,
}
in

let ProfilesContract = {
  schema_version | Positive,
  profiles | Array ModelProfileContract,
}
in

{
  Blake3 = Blake3Contract,
  ModelProfile = ModelProfileContract,
  NixHash = NixHashContract,
  Profiles = ProfilesContract,
  Revision = RevisionContract,
  Precision = PrecisionContract,
  StateKind = StateKindContract,
}
