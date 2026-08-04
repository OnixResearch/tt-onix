# Negative fixture: a zero operation limit cannot bound a real graph.
# Admission must fail with a contract violation before compilation.
let contracts = import "../contracts.ncl" in
let base = import "base-profile.ncl" in
let stripped = std.record.remove "compile_limits" base in
(
  stripped
  & {
    compile_limits = {
      probe_compile_timeout_seconds = 1200,
      full_compile_timeout_seconds = 3600,
      max_stablehlo_operations = 0,
    },
  }
) | contracts.ProfileClosed
  | contracts.Profile
