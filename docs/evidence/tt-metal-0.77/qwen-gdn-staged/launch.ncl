let Run = {
  package | String,
  mode | String,
  entry_point | String,
  timeout_seconds | Number,
} in
{
  runtime | String = "/nix/store/dr9bj8qwflylq39zl5ajcx42cdrb1bv9-tt-metal-0.77.0",
  checkpoint | String = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
  host_threads | Number = 1,
  bf16_decode_override_present | Bool = false,
  sfploadmacro_disabled | Bool = true,
  weights_and_cache_read_only | Bool = true,
  full_component | Run = {
    package = "/nix/store/acqd15fjnx3gcilph1vc80q7qkqw2m9h-qwen36-0.1.0",
    mode = "full",
    entry_point = "qwen36-p150x2-decode-component-probe",
    timeout_seconds = 900,
  },
  full_serving | Run = {
    package = "/nix/store/acqd15fjnx3gcilph1vc80q7qkqw2m9h-qwen36-0.1.0",
    mode = "full",
    entry_point = "qwen36-p150x2-trace-reuse-admission",
    timeout_seconds = 900,
  },
  staged_serving | Run = {
    package = "/nix/store/8h22wxyj3w239r32y2yr2a1dh88yj1ql-qwen36-0.1.0",
    mode = "norm-state",
    entry_point = "qwen36-p150x2-trace-reuse-admission",
    timeout_seconds = 900,
  },
  staged_sequence | Run = {
    package = "/nix/store/jbl3rh5j2gwaw05dgg2msj928gq1f9sw-qwen36-0.1.0",
    mode = "norm-state",
    entry_point = "qwen36-p150x2-decode-component-probe --sequence-steps 64",
    timeout_seconds = 900,
  },
}
