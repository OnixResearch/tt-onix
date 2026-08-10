let Contract = import "contract.ncl" in

let kibibyte = 1024 in
let mebibyte = kibibyte * kibibyte in
let gibibyte = mebibyte * kibibyte in

let default_max_items = 100000 in
let default_max_depth = 64 in
let default_max_fields = 4096 in
let source_max_bytes = mebibyte in
let generated_source_max_bytes = mebibyte in
let portable_evidence_max_bytes = 8 * mebibyte in
let private_log_max_bytes = 128 * mebibyte in
let large_fixture_max_bytes = 8 * mebibyte in
let external_object_max_bytes = 16 * gibibyte in

{
  schema_version = 1,
  parser_limits = [
    {
      family = "json",
      max_bytes = mebibyte,
      max_items = default_max_items,
      max_depth = default_max_depth,
      max_fields = default_max_fields,
    },
    {
      family = "yaml",
      max_bytes = 8 * mebibyte,
      max_items = default_max_items,
      max_depth = default_max_depth,
      max_fields = default_max_fields,
    },
    {
      family = "csv",
      max_bytes = 8 * mebibyte,
      max_items = default_max_items,
      max_depth = 1,
      max_fields = 256,
    },
    {
      family = "binary_frame",
      max_bytes = 16 * mebibyte,
      max_items = default_max_items,
      max_depth = 1,
      max_fields = 256,
    },
    {
      family = "vocabulary",
      max_bytes = 16 * mebibyte,
      max_items = default_max_items,
      max_depth = 1,
      max_fields = 4,
    },
    {
      family = "receipt",
      max_bytes = 2 * mebibyte,
      max_items = default_max_items,
      max_depth = default_max_depth,
      max_fields = default_max_fields,
    },
  ],
  artifact_rules = [
    {
      class = "source",
      max_bytes = source_max_bytes,
      allowed_roots = [".", "cairn", "checks", "crates", "docs", "flake-modules", "model-targets", "nixos", "pkgs", "quality"],
      git_allowed = true,
      blake3_required = false,
      private_content_allowed = false,
    },
    {
      class = "generated_source",
      max_bytes = generated_source_max_bytes,
      allowed_roots = ["model-targets/generated", "quality/generated"],
      git_allowed = true,
      blake3_required = true,
      private_content_allowed = false,
    },
    {
      class = "portable_evidence",
      max_bytes = portable_evidence_max_bytes,
      allowed_roots = ["cairn", "pkgs"],
      git_allowed = true,
      blake3_required = true,
      private_content_allowed = false,
    },
    {
      class = "private_operational_log",
      max_bytes = private_log_max_bytes,
      allowed_roots = [".pi", "generated"],
      git_allowed = false,
      blake3_required = false,
      private_content_allowed = true,
    },
    {
      class = "large_fixture",
      max_bytes = large_fixture_max_bytes,
      allowed_roots = ["pkgs"],
      git_allowed = true,
      blake3_required = true,
      private_content_allowed = false,
    },
    {
      class = "external_large_object",
      max_bytes = external_object_max_bytes,
      allowed_roots = [".pi", "result", "result-1"],
      git_allowed = false,
      blake3_required = true,
      private_content_allowed = false,
    },
  ],
  forbidden_device_paths = ["/dev/tenstorrent", "/dev/hugepages-1G"],
  forbidden_device_commands = ["tt-smi", "tt-flash", "tt-burnin", "ttwkv7-owner-control"],
} | Contract.Policy
