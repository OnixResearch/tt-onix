let contracts = import "../config.ncl" in
{
  schema_version = 1,
  model = {
    model_id = "RWKV/RWKV7-Goose-World2.8-0.1B-HF",
    revision = "d81965cb4e1a9f96696b4f70b84212b8f2e43216",
    checkpoint_blake3 = "905f82048a64b881f9267117a398feb8a8a92bcc5233666bf67904e0d899d0e5",
    layer_count = 12,
    hidden_size = 768,
    head_count = 12,
    head_size = 64,
    vocabulary_size = 65536,
    precision = "fp32_from_bf16",
  },
  tokenizer = {
    vocabulary_blake3 = "3997a74891dd68ced8daadae0d7475274b08988c9263ca042896c8106967aef2",
    vocabulary_entry_count = 65529,
  },
  data = { path = "fixtures/corpus.jsonl", format = "jsonl" },
  sequence = { length = 64, batch = 1 },
  lora = { rank = 2, scale = 1.0, target_roles = [] },
  optimizer = {
    kind = "adamw",
    learning_rate = 0.01,
    gradient_accumulation_steps = 1,
    clip_norm = 1.0,
  },
  seed = 0,
  tolerances = { gradient_agreement = 0.005 },
  output_path = "out/adapter",
  resume = { enabled = false },
} | contracts.ReferenceConfig
