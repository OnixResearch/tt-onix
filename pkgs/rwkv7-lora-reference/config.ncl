let NonEmptyC = std.contract.from_predicate (fun value =>
  std.is_string value && std.string.length value > 0
) in

let PositiveC = std.contract.from_predicate (fun value =>
  std.is_number value && value > 0
) in

let NonNegativeC = std.contract.from_predicate (fun value =>
  std.is_number value && value >= 0
) in

let RevisionC = std.contract.from_predicate (fun value =>
  std.is_string value && std.string.is_match "^[0-9a-f]{40}$" value
) in

let Blake3C = std.contract.from_predicate (fun value =>
  std.is_string value && std.string.is_match "^[0-9a-f]{64}$" value
) in

let admitted_roles = [
  "attention.receptance",
  "attention.key",
  "attention.value",
  "attention.output",
  "feed_forward.key",
  "feed_forward.value",
] in

let LoraRolesC = std.contract.from_predicate (fun value =>
  std.is_array value
  && std.array.length value > 0
  && std.array.all (fun role => std.array.elem role admitted_roles) value
) in

let ModelC = {
  model_id | NonEmptyC,
  revision | RevisionC,
  checkpoint_blake3 | Blake3C,
  layer_count | PositiveC,
  hidden_size | PositiveC,
  head_count | PositiveC,
  head_size | PositiveC,
  vocabulary_size | PositiveC,
  precision | NonEmptyC,
} in

let TokenizerC = {
  vocabulary_blake3 | Blake3C,
  vocabulary_entry_count | PositiveC,
} in

let DataC = {
  path | NonEmptyC,
  format | NonEmptyC,
} in

let SequenceC = {
  length | PositiveC,
  batch | PositiveC,
} in

let LoraC = {
  rank | PositiveC,
  scale | PositiveC,
  target_roles | LoraRolesC,
} in

let OptimizerC = {
  kind | NonEmptyC,
  learning_rate | PositiveC,
  gradient_accumulation_steps | PositiveC,
  clip_norm | PositiveC,
} in

let ResumeC = {
  enabled | Bool,
  adapter_path | NonEmptyC | optional,
} in

let ReferenceConfigC = {
  schema_version | PositiveC,
  model | ModelC,
  tokenizer | TokenizerC,
  data | DataC,
  sequence | SequenceC,
  lora | LoraC,
  optimizer | OptimizerC,
  seed | NonNegativeC,
  tolerances | { _ | PositiveC },
  output_path | NonEmptyC,
  resume | ResumeC,
} in

{
  Blake3 = Blake3C,
  LoraRoles = LoraRolesC,
  NonEmpty = NonEmptyC,
  Positive = PositiveC,
  ReferenceConfig = ReferenceConfigC,
  Revision = RevisionC,
}
