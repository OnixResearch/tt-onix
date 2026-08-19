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

let BoolContract =
  std.contract.from_predicate std.is_bool
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

let NixHash =
  std.contract.from_predicate (fun value =>
    std.is_string value && std.string.is_match "^sha256-[A-Za-z0-9+/]{43}=$" value
  )
in

let HttpsUrl =
  std.contract.from_predicate (fun value =>
    std.is_string value && std.string.is_match "^https://" value
  )
in

let NonEmptyPredicate =
  std.contract.from_predicate (fun value =>
    std.is_array value && std.array.length value > 0
  )
in

let MetadataArtifact = {
  role | std.enum.TagOrString,
  name | NonEmpty,
  url | HttpsUrl,
  byte_count | Positive,
  nix_hash | NixHash,
  blake3 | Blake3,
}
in

let AuthorityContract = {
  schema_version | Positive,
  upstream = {
    repository_url | HttpsUrl,
    branch | NonEmpty,
    commit | Revision,
    base_commit | Revision,
    merged_upstream | BoolContract,
    patch_source | HttpsUrl,
    subtree | NonEmpty,
    adopted_files_count | Positive,
    local_patches | Array NonEmpty,
  },
  pr = {
    number | NonEmpty,
    title | NonEmpty,
    state = "closed",
    merged = false,
    additions | NonNegative,
    deletions | NonNegative,
    changed_files | Positive,
    head_commit | Revision,
  },
  release = {
    model_id | NonEmpty,
    revision | Revision,
    license | NonEmpty,
    architecture | NonEmpty,
    text_model_type | NonEmpty,
    checkpoint_shard_count | Positive,
    checkpoint_shard_names | Array NonEmpty | NonEmptyPredicate,
    hidden_size | Positive,
    layer_count | Positive,
    full_attention_layer_count | Positive,
    linear_attention_layer_count | Positive,
    linear_conv_kernel_dim | Positive,
    linear_key_head_dim | Positive,
    linear_num_key_heads | Positive,
    linear_num_value_heads | Positive,
    linear_value_head_dim | Positive,
    max_position_embeddings | Positive,
    artifacts | Array MetadataArtifact | NonEmptyPredicate,
    weights_excluded = true,
  },
  machine_facts = {
    minimum_mesh | NonEmpty,
    supported_meshes | Array NonEmpty,
    default_weight_precision | NonEmpty,
    vllm = {
      version | NonEmpty,
      mode | NonEmpty,
      multimodal_supported = false,
    },
    measured = {
      traced_4k_tokens_per_s | NonNegative,
      traced_150k_tokens_per_s | NonNegative,
      vllm_e2e_4096_tokens_per_s | NonNegative,
      measured_flags | NonEmpty,
    },
    caveats | Array NonEmpty,
  },
  non_claims | Array NonEmpty,
}
in

{
  Authority = AuthorityContract,
}
