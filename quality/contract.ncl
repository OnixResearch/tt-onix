let NonEmpty = std.contract.from_predicate (fun value =>
  std.is_string value && std.string.length value > 0
) in

let Positive = std.contract.from_predicate (fun value =>
  std.is_number value && value > 0
) in

let ParserFamily = std.contract.from_predicate (fun value =>
  value == "json"
  || value == "yaml"
  || value == "csv"
  || value == "binary_frame"
  || value == "vocabulary"
  || value == "receipt"
) in

let ArtifactClass = std.contract.from_predicate (fun value =>
  value == "source"
  || value == "generated_source"
  || value == "portable_evidence"
  || value == "private_operational_log"
  || value == "large_fixture"
  || value == "external_large_object"
) in

let ParserLimit = {
  family | ParserFamily,
  max_bytes | Positive,
  max_items | Positive,
  max_depth | Positive,
  max_fields | Positive,
} in

let ArtifactRule = {
  class | ArtifactClass,
  max_bytes | Positive,
  allowed_roots | Array NonEmpty,
  git_allowed | Bool,
  blake3_required | Bool,
  private_content_allowed | Bool,
} in

let RepositoryQualityPolicy = {
  schema_version | Positive,
  parser_limits | Array ParserLimit,
  artifact_rules | Array ArtifactRule,
  forbidden_device_paths | Array NonEmpty,
  forbidden_device_commands | Array NonEmpty,
} in

{
  Policy = RepositoryQualityPolicy,
}
