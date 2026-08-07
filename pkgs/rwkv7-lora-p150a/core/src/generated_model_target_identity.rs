// Generated from model-targets/profiles.ncl. Do not edit.
#![allow(dead_code)]

#[rustfmt::skip]
pub const PROFILE_BLAKE3: &str = "d33ed1051c7595327368cce7b42e16f025e8fc78d96addb0353fba2e86314a88";
#[rustfmt::skip]
pub const TARGET_NAME: &str = "rwkv7-goose-world2.8-0.1b-p150x2";
#[rustfmt::skip]
pub const MODEL_ID: &str = "RWKV/RWKV7-Goose-World2.8-0.1B-HF";
#[rustfmt::skip]
pub const MODEL_REVISION: &str = "d81965cb4e1a9f96696b4f70b84212b8f2e43216";
#[rustfmt::skip]
pub const ARCHITECTURE_FAMILY: &str = "rwkv7";
#[rustfmt::skip]
pub const STATE_KIND: &str = "recurrent_matrix";
#[rustfmt::skip]
pub const MODEL_PRECISION: &str = "bf16_checkpoint_fp32_reference";
#[rustfmt::skip]
pub const ARCHITECTURE_ADAPTER_ID: &str = "rwkv7-metalium-p150x2-v1";
#[rustfmt::skip]
pub const EVIDENCE_ADAPTER_ID: &str = "rwkv7-p150x2-evidence-v1";
#[rustfmt::skip]
pub const TOPOLOGY_ID: &str = "p150x2-exact-physical-0-1";
#[rustfmt::skip]
pub const REPLICATION: &str = "full-lockstep";
#[rustfmt::skip]
pub const CPU_REFERENCE_PRECISION: &str = "fp32_from_bf16";
#[rustfmt::skip]
pub const RUNTIME_ARITHMETIC_PRECISION: &str = "cpu_fp32_from_bf16";
#[rustfmt::skip]
pub const TTWKV7_BOUNDARY_PRECISION: &str = "little_endian_bf16_storage_cpu_fp32_recurrence";
#[rustfmt::skip]
pub const ADAPTER_FORMAT: &str = "safetensors-adapter-only";
#[rustfmt::skip]
pub const MODEL_BLAKE3: &str = "905f82048a64b881f9267117a398feb8a8a92bcc5233666bf67904e0d899d0e5";
#[rustfmt::skip]
pub const MODEL_SHA256_SRI: &str = "sha256-uWqL3CHhX3HgyVZT3MO+ieVkthmtUHPJ7b+9B/eElFM=";
#[rustfmt::skip]
pub const TOKENIZER_VOCAB_BLAKE3: &str = "3997a74891dd68ced8daadae0d7475274b08988c9263ca042896c8106967aef2";
#[rustfmt::skip]
pub const TOKENIZER_VOCAB_SHA256_SRI: &str = "sha256-5t7j1OMbTVxArJlQisbHAc7vS+1oG/IWfOmpCFUryok=";
#[rustfmt::skip]
pub const TOKENIZER_CONFIG_BLAKE3: &str = "b2411eb362aefa260493811c9414e8da589a19d6cec44e8456953507e293755e";
#[rustfmt::skip]
pub const TOKENIZER_CONFIG_SHA256_SRI: &str = "sha256-TgOqD11rGkAGoNnp8HDwFBjnNKsXx8SPKDM1lta9Xik=";
#[rustfmt::skip]
pub const ADDED_TOKENS_BLAKE3: &str = "02893c22a1e92502fdd31ba4d57b6e574692023505be7ba82d68d1e3142ff02f";
#[rustfmt::skip]
pub const ADDED_TOKENS_SHA256_SRI: &str = "sha256-o0nK5s2qaAz2/A0pKbFvKp7bQ+twJ7LjRLHKgGOFT7k=";
#[rustfmt::skip]
pub const TOKENIZER_IMPLEMENTATION_BLAKE3: &str = "0a2a88e97b455858e03bbbc83bb0228d8f36c2731fcb91cd94f05e2930e2aa24";
#[rustfmt::skip]
pub const TOKENIZER_IMPLEMENTATION_SHA256_SRI: &str = "sha256-qspeag9W0EPKFlTp3K+Qb888DgO1FyhjrXUGDoaFoQ4=";
#[rustfmt::skip]
pub const SPECIAL_TOKENS_MAP_BLAKE3: &str = "751ae3ea4b59073218a85facbee1536739b0aa26d5d5670e11ef6815e5bac870";
#[rustfmt::skip]
pub const SPECIAL_TOKENS_MAP_SHA256_SRI: &str = "sha256-H1EppN7ADOM+XFzScmyVKyKkeCcyHu08UY1DXUpkYBU=";
#[rustfmt::skip]
pub const MODEL_CONFIG_BLAKE3: &str = "113edfd55813d327ae7e37987ee9c5ed123c69fa809670b3a0bcc07fd1e9295d";
#[rustfmt::skip]
pub const MODEL_CONFIG_SHA256_SRI: &str = "sha256-VcFZ/IlA4WVXpCsE8K7QN0TxdiIcwSY3aUqx9+EMTG8=";
#[rustfmt::skip]
pub const GENERATION_CONFIG_BLAKE3: &str = "2288838a56ea704a85691828bbc7f0ab2934c949b13a4d8f3af1b13d955ac2fa";
#[rustfmt::skip]
pub const GENERATION_CONFIG_SHA256_SRI: &str = "sha256-2milZURvylpqKvSzCIkS6VOWX7+m6WgBmhTXVj1Y2Tc=";
pub const MODEL_BYTE_COUNT: u64 = 382111072;
pub const MODEL_LAYER_COUNT: usize = 12;
pub const HIDDEN_SIZE: usize = 768;
pub const VOCABULARY_SIZE: usize = 65536;
pub const MAXIMUM_PROMPT_TOKENS: usize = 32;
pub const PER_LAYER_STATE_ELEMENT_COUNT: usize = 50688;
pub const HEAD_COUNT: usize = 12;
pub const HEAD_SIZE: usize = 64;
pub const INTERMEDIATE_SIZE: usize = 3072;
pub const COMPLETE_STATE_ELEMENT_COUNT: usize = 608256;
pub const HARNESS_ORACLE_TOLERANCE: f32 = 1e-5;
pub const HARNESS_REPLAY_TOLERANCE: f32 = 1e-5;
pub const STATE_CARRY_DIVERGENCE_FLOOR: f32 = 0.0001;
pub const FINAL_HIDDEN_TOLERANCE: f32 = 0.5;
pub const LOGIT_TOLERANCE: f32 = 1.0;
pub const RECURRENT_STATE_TOLERANCE: f32 = 1.0;
pub const CONTROL_DIVERGENCE_FLOOR: f32 = 0.5;
pub const LORA_GRADIENT_AGREEMENT_TOLERANCE: f32 = 0.005;
pub const LORA_FINITE_DIFFERENCE_TOLERANCE: f32 = 0.02;
#[rustfmt::skip]
pub const ARTIFACT_BLAKE3_JSON: &str = r#"{"checkpoint":"905f82048a64b881f9267117a398feb8a8a92bcc5233666bf67904e0d899d0e5","generation_config":"2288838a56ea704a85691828bbc7f0ab2934c949b13a4d8f3af1b13d955ac2fa","model_config":"113edfd55813d327ae7e37987ee9c5ed123c69fa809670b3a0bcc07fd1e9295d","tokenizer_added_tokens":"02893c22a1e92502fdd31ba4d57b6e574692023505be7ba82d68d1e3142ff02f","tokenizer_config":"b2411eb362aefa260493811c9414e8da589a19d6cec44e8456953507e293755e","tokenizer_implementation":"0a2a88e97b455858e03bbbc83bb0228d8f36c2731fcb91cd94f05e2930e2aa24","tokenizer_special_tokens":"751ae3ea4b59073218a85facbee1536739b0aa26d5d5670e11ef6815e5bac870","tokenizer_vocabulary":"3997a74891dd68ced8daadae0d7475274b08988c9263ca042896c8106967aef2"}"#;
#[rustfmt::skip]
pub const MESH_SHAPE_JSON: &str = r#"[1,2]"#;
#[rustfmt::skip]
pub const PHYSICAL_DEVICE_IDS_JSON: &str = r#"[0,1]"#;
#[rustfmt::skip]
pub const PROFILES_JSON: &str = r#"[{"adapters":{"architecture":"rwkv7-metalium-p150x2-v1","evidence":"rwkv7-p150x2-evidence-v1"},"artifacts":[{"blake3":"905f82048a64b881f9267117a398feb8a8a92bcc5233666bf67904e0d899d0e5","name":"rwkv7-goose-world2.8-0.1b-d81965cb4e1a9f96696b4f70b84212b8f2e43216.safetensors","nix_hash":"sha256-uWqL3CHhX3HgyVZT3MO+ieVkthmtUHPJ7b+9B/eElFM=","role":"checkpoint","url":"https://huggingface.co/RWKV/RWKV7-Goose-World2.8-0.1B-HF/resolve/d81965cb4e1a9f96696b4f70b84212b8f2e43216/model.safetensors"},{"blake3":"3997a74891dd68ced8daadae0d7475274b08988c9263ca042896c8106967aef2","name":"rwkv_vocab_v20230424.txt","nix_hash":"sha256-5t7j1OMbTVxArJlQisbHAc7vS+1oG/IWfOmpCFUryok=","role":"tokenizer_vocabulary","url":"https://huggingface.co/RWKV/RWKV7-Goose-World2.8-0.1B-HF/resolve/d81965cb4e1a9f96696b4f70b84212b8f2e43216/rwkv_vocab_v20230424.txt"},{"blake3":"b2411eb362aefa260493811c9414e8da589a19d6cec44e8456953507e293755e","name":"tokenizer_config.json","nix_hash":"sha256-TgOqD11rGkAGoNnp8HDwFBjnNKsXx8SPKDM1lta9Xik=","role":"tokenizer_config","url":"https://huggingface.co/RWKV/RWKV7-Goose-World2.8-0.1B-HF/resolve/d81965cb4e1a9f96696b4f70b84212b8f2e43216/tokenizer_config.json"},{"blake3":"02893c22a1e92502fdd31ba4d57b6e574692023505be7ba82d68d1e3142ff02f","name":"added_tokens.json","nix_hash":"sha256-o0nK5s2qaAz2/A0pKbFvKp7bQ+twJ7LjRLHKgGOFT7k=","role":"tokenizer_added_tokens","url":"https://huggingface.co/RWKV/RWKV7-Goose-World2.8-0.1B-HF/resolve/d81965cb4e1a9f96696b4f70b84212b8f2e43216/added_tokens.json"},{"blake3":"0a2a88e97b455858e03bbbc83bb0228d8f36c2731fcb91cd94f05e2930e2aa24","name":"hf_rwkv_tokenizer.py","nix_hash":"sha256-qspeag9W0EPKFlTp3K+Qb888DgO1FyhjrXUGDoaFoQ4=","role":"tokenizer_implementation","url":"https://huggingface.co/RWKV/RWKV7-Goose-World2.8-0.1B-HF/resolve/d81965cb4e1a9f96696b4f70b84212b8f2e43216/hf_rwkv_tokenizer.py"},{"blake3":"751ae3ea4b59073218a85facbee1536739b0aa26d5d5670e11ef6815e5bac870","name":"special_tokens_map.json","nix_hash":"sha256-H1EppN7ADOM+XFzScmyVKyKkeCcyHu08UY1DXUpkYBU=","role":"tokenizer_special_tokens","url":"https://huggingface.co/RWKV/RWKV7-Goose-World2.8-0.1B-HF/resolve/d81965cb4e1a9f96696b4f70b84212b8f2e43216/special_tokens_map.json"},{"blake3":"113edfd55813d327ae7e37987ee9c5ed123c69fa809670b3a0bcc07fd1e9295d","name":"config.json","nix_hash":"sha256-VcFZ/IlA4WVXpCsE8K7QN0TxdiIcwSY3aUqx9+EMTG8=","role":"model_config","url":"https://huggingface.co/RWKV/RWKV7-Goose-World2.8-0.1B-HF/resolve/d81965cb4e1a9f96696b4f70b84212b8f2e43216/config.json"},{"blake3":"2288838a56ea704a85691828bbc7f0ab2934c949b13a4d8f3af1b13d955ac2fa","name":"generation_config.json","nix_hash":"sha256-2milZURvylpqKvSzCIkS6VOWX7+m6WgBmhTXVj1Y2Tc=","role":"generation_config","url":"https://huggingface.co/RWKV/RWKV7-Goose-World2.8-0.1B-HF/resolve/d81965cb4e1a9f96696b4f70b84212b8f2e43216/generation_config.json"}],"consumer":{"adapter_format":"safetensors-adapter-only","dimensions":{"complete_state_element_count":608256,"head_count":12,"head_size":64,"intermediate_size":3072},"precisions":{"cpu_reference":"fp32_from_bf16","runtime_arithmetic":"cpu_fp32_from_bf16","ttwkv7_boundary":"little_endian_bf16_storage_cpu_fp32_recurrence"},"tolerances":{"control_divergence_floor":0.5,"final_hidden":0.5,"harness_oracle":0.00001,"harness_replay":0.00001,"logits":1,"lora_finite_difference":0.02,"lora_gradient_agreement":0.005,"recurrent_state":1,"state_carry_divergence_floor":0.0001}},"model":{"architecture_family":"rwkv7","checkpoint_layout":"safetensors_single_file","id":"RWKV/RWKV7-Goose-World2.8-0.1B-HF","license":"Apache-2.0","precision":"bf16_checkpoint_fp32_reference","revision":"d81965cb4e1a9f96696b4f70b84212b8f2e43216"},"profile_blake3":"d33ed1051c7595327368cce7b42e16f025e8fc78d96addb0353fba2e86314a88","resources":{"checkpoint_bytes":382111072,"hidden_size":768,"layer_count":12,"maximum_prompt_tokens":32,"vocabulary_size":65536},"runnable":true,"state":{"carry_semantics":"token_shift_and_wkv_matrix_carry","kind":"recurrent_matrix","per_layer_element_count":50688},"target_name":"rwkv7-goose-world2.8-0.1b-p150x2","topology":{"id":"p150x2-exact-physical-0-1","mesh_shape":[1,2],"physical_device_ids":[0,1],"replication":"full-lockstep"}},{"adapters":{"architecture":"synthetic-transformer-device-free-v1","evidence":"synthetic-transformer-evidence-v1"},"artifacts":[{"blake3":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","name":"synthetic-transformer-weights.bin","nix_hash":"sha256-uWqL3CHhX3HgyVZT3MO+ieVkthmtUHPJ7b+9B/eElFM=","role":"fixture_weights","url":"fixture://synthetic-transformer-weights"}],"consumer":{"adapter_format":"fixture-only","dimensions":{"complete_state_element_count":256,"head_count":2,"head_size":32,"intermediate_size":128},"precisions":{"cpu_reference":"fp32_fixture","runtime_arithmetic":"fp32_fixture","ttwkv7_boundary":"not_applicable"},"tolerances":{"control_divergence_floor":0.01,"final_hidden":0.01,"harness_oracle":0.00001,"harness_replay":0.00001,"logits":0.01,"lora_finite_difference":0.02,"lora_gradient_agreement":0.005,"recurrent_state":0.01,"state_carry_divergence_floor":0.0001}},"model":{"architecture_family":"synthetic_transformer","checkpoint_layout":"synthetic_single_file","id":"fixture/synthetic-transformer","license":"fixture-only","precision":"bf16_fixture","revision":"1111111111111111111111111111111111111111"},"profile_blake3":"0b3fdfe249b30b320be34ece4debfa3107ab61cdbec46ef304957c1a72e2f114","resources":{"checkpoint_bytes":1024,"hidden_size":64,"layer_count":2,"maximum_prompt_tokens":8,"vocabulary_size":256},"runnable":false,"state":{"carry_semantics":"append_key_and_value_rows","kind":"kv_cache","per_layer_element_count":128},"target_name":"synthetic-transformer-kv-cache-fixture","topology":{"id":"device-free-fixture","mesh_shape":[1,1],"physical_device_ids":[0],"replication":"none"}}]"#;
