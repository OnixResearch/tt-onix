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

let MetadataArtifact = {
  role | NonEmpty,
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
    merge_parent | Revision,
    tt_metal_base_commit | Revision,
    subtree | NonEmpty,
    subtree_file_count | Positive,
    tree_identity_algorithm | NonEmpty,
    tree_blake3 | Blake3,
  },
  release = {
    model_id | NonEmpty,
    revision | Revision,
    license | NonEmpty,
    architecture | NonEmpty,
    text_model_type | NonEmpty,
    config_transformers_version | NonEmpty,
    checkpoint_shard_count | Positive,
    checkpoint_shard_names | Array NonEmpty,
    checkpoint_tensor_count | Positive,
    checkpoint_bytes | Positive,
    artifacts | Array MetadataArtifact,
  },
  machine_facts = {
    firmware_minimum | NonEmpty,
    known_good_firmware | NonEmpty,
    python_env_transformers | NonEmpty,
    python_env_ttnn | NonEmpty,
    python_env_ttl | NonEmpty,
    cpu_reference_transformers | NonEmpty,
    naming_mismatch | NonEmpty,
  },
}
in

{
  Authority = AuthorityContract,
}
