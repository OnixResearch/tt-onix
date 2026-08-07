let contracts = import "config.ncl" in
let target_profiles = import "../../model-targets/profiles.ncl" in
let target = std.array.first (std.array.filter
  (fun profile => profile.target_name == "rwkv7-goose-world2.8-0.1b-p150x2")
  target_profiles.profiles) in
let artifact = fun role => std.array.first (std.array.filter
  (fun value => value.role == role)
  target.artifacts) in
{
  schema_version = 1,
  model = {
    model_id = target.model.id,
    revision = target.model.revision,
    checkpoint_blake3 = (artifact "checkpoint").blake3,
    layer_count = target.resources.layer_count,
    hidden_size = target.resources.hidden_size,
    head_count = target.consumer.dimensions.head_count,
    head_size = target.consumer.dimensions.head_size,
    vocabulary_size = target.resources.vocabulary_size,
    precision = target.consumer.precisions.cpu_reference,
  },
  tokenizer = {
    vocabulary_blake3 = (artifact "tokenizer_vocabulary").blake3,
    vocabulary_entry_count = 65529,
  },
  data = {
    path = "fixtures/corpus.jsonl",
    format = "jsonl",
  },
  sequence = {
    length = 64,
    batch = 1,
  },
  lora = {
    rank = 2,
    scale = 1.0,
    target_roles = [
      "attention.receptance",
      "attention.key",
      "attention.value",
      "attention.output",
      "feed_forward.key",
      "feed_forward.value",
    ],
  },
  optimizer = {
    kind = "adamw",
    learning_rate = 0.01,
    gradient_accumulation_steps = 1,
    clip_norm = 1.0,
  },
  seed = 0,
  tolerances = {
    gradient_agreement = target.consumer.tolerances.lora_gradient_agreement,
    finite_difference = target.consumer.tolerances.lora_finite_difference,
    equation = 0.001,
  },
  output_path = "out/adapter",
  resume = {
    enabled = false,
  },
} | contracts.ReferenceConfig
