let NonEmpty = std.contract.from_predicate (fun value =>
  std.is_string value && std.string.length value > 0
) in

let Positive = std.contract.from_predicate (fun value =>
  std.is_number value && value > 0
) in

let NonNegative = std.contract.from_predicate (fun value =>
  std.is_number value && value >= 0
) in

let RevisionContract = std.contract.from_predicate (fun value =>
  std.is_string value && std.string.is_match "^[0-9a-f]{40}$" value
) in

let Blake3Contract = std.contract.from_predicate (fun value =>
  std.is_string value && std.string.is_match "^[0-9a-f]{64}$" value
) in

let NixHashContract = std.contract.from_predicate (fun value =>
  std.is_string value && std.string.is_match "^sha256-[A-Za-z0-9+/]{43}=$" value
) in

let Url = std.contract.from_predicate (fun value =>
  std.is_string value
  && (std.string.is_match "^https://" value || std.string.is_match "^fixture://" value)
) in

let StateKindContract = std.contract.from_predicate (fun value =>
  value == "recurrent_matrix" || value == "kv_cache"
) in

let ProfileArtifact = {
  role | NonEmpty,
  name | NonEmpty,
  url | Url,
  nix_hash | NixHashContract,
  blake3 | Blake3Contract,
} in

let ModelProfileContract = {
  target_name | NonEmpty,
  runnable | Bool,
  model | {
    id | NonEmpty,
    revision | RevisionContract,
    license | NonEmpty,
    architecture_family | NonEmpty,
    checkpoint_layout | NonEmpty,
    precision | NonEmpty,
  },
  state | {
    kind | StateKindContract,
    per_layer_element_count | Positive,
    carry_semantics | NonEmpty,
  },
  resources | {
    checkpoint_bytes | Positive,
    layer_count | Positive,
    hidden_size | Positive,
    vocabulary_size | Positive,
    maximum_prompt_tokens | Positive,
  },
  adapters | {
    architecture | NonEmpty,
    evidence | NonEmpty,
  },
  topology | {
    id | NonEmpty,
    mesh_shape | Array Positive,
    physical_device_ids | Array NonNegative,
    replication | NonEmpty,
  },
  artifacts | Array ProfileArtifact,
} in

let ProfilesContract = {
  schema_version | Positive,
  profiles | Array ModelProfileContract,
} in

{
  Blake3 = Blake3Contract,
  ModelProfile = ModelProfileContract,
  NixHash = NixHashContract,
  Profiles = ProfilesContract,
  Revision = RevisionContract,
  StateKind = StateKindContract,
}
