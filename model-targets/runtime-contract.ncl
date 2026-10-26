# SPDX-License-Identifier: LGPL-2.1-only
# r[impl tenstorrent.runtime_profile.registry]
# r[impl tenstorrent.runtime_profile.contract]
#
# Closed, typed contract for the runtime-profile registry. A profile says how one
# model runtime is launched: its package, entry point, argv, environment, mesh,
# checkpoint authority, cache, admission kind, readiness signal, and reset policy.
#
# The registry is data. It never evaluates a numerical gate, never computes an
# admission verdict, and never re-serializes a model's own receipt schema. Those
# stay with the model packages; see `model-targets/runtime-profiles.ncl` for the
# citations behind each recorded value.
#
# `@package@` is the only permitted placeholder in an argv token. It MUST be the
# token's prefix and expands to the profile package's store path, which keeps a
# profile from hardcoding a store hash that a rebuild would move.
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

let NonNegative =
  std.contract.from_predicate (fun value =>
    std.is_number value && value >= 0
  )
in

let Blake3 =
  std.contract.from_predicate (fun value =>
    std.is_string value && std.string.is_match "^[0-9a-f]{64}$" value
  )
in

let Revision =
  std.contract.from_predicate (fun value =>
    std.is_string value && std.string.is_match "^[0-9a-f]{40}$" value
  )
in

# Custom contracts keep every refusal greppable. The registry check asserts the
# message of each negative fixture, so a generic "contract broken" is not enough.
let Enum =
  fun failure allowed =>
    std.contract.custom (fun label value =>
      if std.is_string value && std.array.any (fun candidate => candidate == value) allowed then
        'Ok value
      else
        # Nickel records are recursive: the field value must not share the
        # parameter's name, so the message parameter is called `failure`.
        'Error { message = failure }
    )
in

let Require =
  fun failure predicate =>
    std.contract.custom (fun label value =>
      if predicate value then 'Ok value else 'Error { message = failure }
    )
in

let Schema = Enum "runtime profile schema must be onix.runtime-profile.v1" [ "onix.runtime-profile.v1" ] in

let Mode = Enum "runtime profile mode must be serve or job" [ "serve", "job" ] in

let AdmissionKind =
  Enum "runtime profile admission kind must be receipt-set, authority-digest, or gate-exit" [
    "receipt-set",
    "authority-digest",
    "gate-exit",
  ]
in

let ReadinessKind =
  Enum "runtime profile readiness kind must be http-healthz, exit-code, or self-test" [
    "http-healthz",
    "exit-code",
    "self-test",
  ]
in

let ResetPolicy =
  Enum "runtime profile reset policy must be never, required-before-native, or always" [
    "never",
    "required-before-native",
    "always",
  ]
in

let ProfileName =
  std.contract.from_predicate (fun value =>
    std.is_string value && std.string.is_match "^[a-z0-9][a-z0-9-]*$" value
  )
in

let Package = {
  attribute | NonEmpty,
  store_name | NonEmpty,
}
in

let Entry = {
  # Path to the entry wrapper, relative to the profile package's store path,
  # for example `bin/qwen38-p150x2-serve`.
  binary | NonEmpty,
  # Arguments the launcher must place after the binary, verbatim. `@package@`
  # is the only permitted placeholder.
  argv_prefix | Array NonEmpty,
}
in

let Mesh = {
  name | NonEmpty,
  shape | Array Positive,
  physical_device_ids | Array NonNegative,
  # Free text naming the fabric or mesh-graph descriptor the runtime requires.
  # The registry records it; it does not interpret it.
  fabric | NonEmpty,
}
in

let Limits = {
  max_sequence_length | Positive,
  max_generation_tokens | Positive,
}
in

let Checkpoint = {
  # Absent when the artifact is materialized by Nix from a pinned input instead
  # of living at a host path. Absence is explicit; an empty string is not allowed.
  path | optional | NonEmpty,
  revision | Revision,
  authority_kind | NonEmpty,
  # Repository-relative path of the authority document `authority_blake3` hashes.
  # Present only for `generated-authority-blake3`, where the check recomputes it.
  authority_path | optional | NonEmpty,
  authority_blake3 | Blake3,
}
in

let Cache = {
  # Absent when the runtime declares no cache root and uses its own default.
  root | NonEmpty,
  # Absent when the runtime keys its cache by the root alone.
  key | optional | NonEmpty,
}
in

let Precision = {
  # Absent when the model has no named precision preset.
  preset | optional | NonEmpty,
  # Environment keys that select precision or fusion behaviour. Every name must
  # exist in `env`, so the profile keeps one copy of each value.
  flags | Array NonEmpty,
}
in

let Receipt = {
  kind | NonEmpty,
  path | NonEmpty,
  blake3 | Blake3,
}
in

# One gate a gate-exit profile runs. `artifact` is package-relative, and its
# digest binds the recorded gate name to the installed program, not to a gate
# result. Both are optional and must appear together: a sibling gate wrapper that
# lives in another derivation cannot be named package-relatively, and an
# absolute store path would go stale on every rebuild.
let Gate = {
  name | NonEmpty,
  artifact | optional | NonEmpty,
  blake3 | optional | Blake3,
}
in

let Admission = {
  kind | AdmissionKind,
  receipts | optional | Array Receipt,
  gates | optional | Array Gate,
}
in

# The runtime a profile launches, as a package-relative root. Absent when the
# package is its own runtime (the native stack) or when the runtime lives outside
# the package (the TT-NN RWKV-7 wrapper points TT_METAL_HOME at a symlinkJoin).
let Runtime = {
  root | NonEmpty,
  # Package-relative symlink naming the package's execution identity, when the
  # package exports one.
  execution_identity | optional | NonEmpty,
}
in

let Readiness = {
  kind | ReadinessKind,
  path | optional | NonEmpty,
  timeout_seconds | optional | Positive,
}
in

let Reset = {
  on_release | ResetPolicy,
  device_paths | Array NonEmpty,
}
in

let RuntimeProfile = {
  schema | Schema,
  name | ProfileName,
  mode | Mode,
  package | Package,
  entry | Entry,
  env | { _ | String },
  mesh | Mesh,
  port | optional | Positive,
  limits | Limits,
  checkpoint | Checkpoint,
  cache | optional | Cache,
  precision | Precision,
  admission | Admission,
  readiness | Readiness,
  reset | Reset,
  runtime | optional | Runtime,
  model_target_name | optional | NonEmpty,
  model_target_blake3 | optional | Blake3,
  non_claims | Array NonEmpty,
}
in

# Cross-field invariants as a plain function, so the same rules apply to a single
# profile and to the whole registry without composing record contracts (which
# would force their field definitions at import time).
let semanticErrors = fun profile =>
  let derived = {
    ids = profile.mesh.physical_device_ids,
    device_count = std.array.fold_left (fun total size => total * size) 1 profile.mesh.shape,
    device_paths = std.array.map (fun id => "/dev/tenstorrent/%{std.to_string id}") ids,
    env_keys = std.record.fields profile.env,
    placeholder_is_prefix =
      std.array.all
        (fun token =>
          if std.string.is_match "@package@" token then
            std.string.is_match "^@package@[^@]*$" token
          else
            true
        )
        profile.entry.argv_prefix,
  } in
  let checks = [
    {
      ok = derived.device_count == std.array.length derived.ids,
      message = "runtime profile mesh shape must multiply to the declared physical device id count",
    },
    {
      ok = std.array.length derived.ids > 0,
      message = "runtime profile mesh must declare at least one physical device id",
    },
    {
      ok = std.array.all (fun id => id == 0 || id == 1) derived.ids,
      message = "runtime profile mesh physical device ids must be 0 or 1",
    },
    {
      ok = profile.limits.max_generation_tokens <= profile.limits.max_sequence_length,
      message = "runtime profile max_generation_tokens must not exceed max_sequence_length",
    },
    {
      ok = std.array.all
        (fun flag => std.array.any (fun key => key == flag) derived.env_keys)
        profile.precision.flags,
      message = "runtime profile precision flags must name environment keys",
    },
    {
      ok = std.array.all
        (fun path => std.array.any (fun device => device == path) derived.device_paths)
        profile.reset.device_paths,
      message = "runtime profile reset device paths must be the devices the profile declares",
    },
    {
      ok = derived.placeholder_is_prefix,
      message = "runtime profile argv @package@ placeholder must prefix its token",
    },
    {
      ok = std.record.has_field "model_target_name" profile
      == std.record.has_field "model_target_blake3" profile,
      message = "runtime profile model target name and identity must appear together",
    },
    {
      ok = std.array.all
        (fun gate => std.record.has_field "artifact" gate == std.record.has_field "blake3" gate)
        (if std.record.has_field "gates" profile.admission then profile.admission.gates else []),
      message = "runtime profile gate artifact and digest must appear together",
    },
  ]
  @ (
    if profile.mode == "serve" then
      [
        {
          ok = std.record.has_field "port" profile,
          message = "serve runtime profile must declare a port",
        },
        {
          ok = profile.mesh.shape == [ 1, 2 ],
          message = "serve runtime profile mesh shape must be exactly [1,2]",
        },
        {
          ok = derived.ids == [ 0, 1 ],
          message = "serve runtime profile mesh physical device ids must be exactly [0,1]",
        },
        {
          ok = profile.readiness.kind == "http-healthz",
          message = "serve runtime profile readiness must be http-healthz",
        },
      ]
    else
      [
        {
          ok = !(std.record.has_field "port" profile),
          message = "job runtime profile must not declare a port",
        },
        {
          ok = profile.readiness.kind != "http-healthz",
          message = "job runtime profile readiness must not be http-healthz",
        },
      ]
  )
  @ (
    if profile.readiness.kind == "http-healthz" then
      [
        {
          ok = std.record.has_field "path" profile.readiness,
          message = "http-healthz readiness requires a health path",
        },
      ]
    else
      [
        {
          ok = !(std.record.has_field "path" profile.readiness),
          message = "non-http readiness must not declare a health path",
        },
      ]
  )
  @ (
    if profile.admission.kind == "receipt-set" then
      [
        {
          ok = std.record.has_field "receipts" profile.admission
          && std.array.length profile.admission.receipts > 0,
          message = "receipt-set admission requires at least one receipt",
        },
        {
          ok = !(std.record.has_field "gates" profile.admission),
          message = "receipt-set admission must not declare gates",
        },
      ]
    else if profile.admission.kind == "gate-exit" then
      [
        {
          ok = std.record.has_field "gates" profile.admission
          && std.array.length profile.admission.gates > 0,
          message = "gate-exit admission requires at least one gate",
        },
        {
          ok = !(std.record.has_field "receipts" profile.admission),
          message = "gate-exit admission must not declare receipts",
        },
      ]
    else
      [
        {
          ok = !(std.record.has_field "receipts" profile.admission),
          message = "authority-digest admission must not declare receipts",
        },
        {
          ok = !(std.record.has_field "gates" profile.admission),
          message = "authority-digest admission must not declare gates",
        },
      ]
  )
  @ (
    if profile.checkpoint.authority_kind == "generated-authority-blake3" then
      [
        {
          ok = std.record.has_field "path" profile.checkpoint,
          message = "generated-authority-blake3 checkpoint must declare its host path",
        },
        {
          ok = std.record.has_field "authority_path" profile.checkpoint,
          message = "generated-authority-blake3 checkpoint must declare its authority document",
        },
      ]
    else
      [
        {
          ok = !(std.record.has_field "authority_path" profile.checkpoint),
          message = "non-generated authority must not declare an authority document",
        },
      ]
  ) in
  std.array.map (fun check => check.message) (std.array.filter (fun check => !check.ok) checks)
in

# One profile: the contract's second layer.
let ProfileSemantics = std.contract.custom (fun label profile =>
  let failures = semanticErrors profile in
  if failures == [] then 'Ok profile else 'Error { message = std.array.first failures }
) in

# The whole registry: the record shape, then semantics for every profile, then
# unique names. Callers apply the three in order.
let RuntimeProfilesShape = {
  schema_version | Positive,
  profiles | Array RuntimeProfile,
} in

let ProfilesSemantics = std.contract.custom (fun label value =>
  let failures = std.array.flatten (std.array.map (fun profile => semanticErrors profile) value.profiles) in
  if failures == [] then 'Ok value else 'Error { message = std.array.first failures }
) in

let UniqueNames = Require
  "runtime profile names must be unique"
  (fun value =>
    let derived = {
      names = std.array.map (fun profile => profile.name) value.profiles,
      distinct = std.array.fold_left
        (fun seen name =>
          if std.array.any (fun other => other == name) seen then seen else seen @ [ name ])
        []
        names,
    } in
    std.array.length derived.names == std.array.length derived.distinct)
in

{
  # Public entry points. Field names differ from the bindings because Nickel
  # records are recursive: `Profiles = Profiles` would self-refer.
  Profile = RuntimeProfile,
  ProfileSemantics = ProfileSemantics,
  Profiles = RuntimeProfilesShape,
  Semantics = ProfilesSemantics,
  Unique = UniqueNames,
}
