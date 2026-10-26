# SPDX-License-Identifier: LGPL-2.1-only
# r[verify tenstorrent.runtime_profile.registry.fresh]
# Device-free check for the runtime-profile registry: the committed export is
# byte-identical to its Nickel source, every negative fixture is refused with its
# named rule, and every recorded authority or receipt digest is the file's own.
{
  lib,
  pkgs,
  nickel,
  b3sum,
  python3,
  modelTargets,
}:
let
  data = builtins.fromJSON (builtins.readFile ./generated/runtime-profiles.json);
  repositorySource = lib.cleanSource ../.;
  names = map (profile: profile.name) data.profiles;
  expectedNames = [
    "metalium-rs-rwkv7"
    "nanbeige43"
    "qwen38"
    "rwkv7-ttnn"
  ];
  profile =
    name:
    let
      found = lib.filter (candidate: candidate.name == name) data.profiles;
    in
    if lib.length found == 1 then
      lib.head found
    else
      throw "runtime profile '${name}' must appear exactly once";
  rwkvTarget = modelTargets.getProfile "rwkv7-goose-world2.8-0.1b-p150x2";
  rwkvIdentity = rwkvTarget.profile_identity;
  modelTargetReferences = lib.filter (entry: entry ? model_target_name) data.profiles;
  modelTargetReferenceMismatch = lib.filter (
    entry:
    (modelTargets.getProfile entry.model_target_name).profile_identity != entry.model_target_blake3
  ) modelTargetReferences;
  # A fixture is a profile mutation the contract must refuse, paired with the
  # message it must print. Each message names a rule in runtime-contract.ncl.
  fixtures = [
    {
      file = "invalid-runtime-unknown-field.ncl";
      message = "extra field `undocumented_launch_flag`";
    }
    {
      file = "invalid-runtime-missing-field.ncl";
      message = "missing definition for `mode`";
    }
    {
      file = "invalid-runtime-mode.ncl";
      message = "runtime profile mode must be serve or job";
    }
    {
      file = "invalid-runtime-authority-blake3.ncl";
      message = "contract broken by the value of `authority_blake3`";
    }
    {
      file = "invalid-runtime-serve-device-ids.ncl";
      message = "serve runtime profile mesh physical device ids must be exactly [0,1]";
    }
    {
      file = "invalid-runtime-serve-without-port.ncl";
      message = "serve runtime profile must declare a port";
    }
    {
      file = "invalid-runtime-duplicate-name.ncl";
      message = "runtime profile names must be unique";
    }
  ];
  # The registry readers live in one file so the shell below stays declarative.
  verify = pkgs.writeText "verify-runtime-profile-registry.py" ''
    import json
    import os
    import sys


    def load(path):
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)


    def authorities(registry):
        for entry in registry["profiles"]:
            checkpoint = entry["checkpoint"]
            if "authority_path" in checkpoint:
                print(checkpoint["authority_path"], checkpoint["authority_blake3"])


    def receipts(registry):
        for entry in registry["profiles"]:
            for receipt in entry.get("admission", {}).get("receipts", []):
                path = receipt["path"]
                if not os.path.isfile(path):
                    raise SystemExit(f"{path}: missing receipt")
                document = load(path)
                if document["admission"]["admitted"] is not True:
                    raise SystemExit(f"{path}: receipt is not admitted")
                if document["receipt_kind"] != receipt["kind"]:
                    raise SystemExit(f"{path}: receipt kind mismatch")
                print(path, receipt["blake3"])


    def gates(registry):
        for entry in registry["profiles"]:
            for gate in entry.get("admission", {}).get("gates", []):
                if "artifact" in gate:
                    print(entry["name"], gate["artifact"])


    def model_targets(registry, expected):
        referenced = {
            entry["name"]: entry["model_target_blake3"]
            for entry in registry["profiles"]
            if "model_target_blake3" in entry
        }
        if not referenced:
            raise SystemExit("no runtime profile references a model-target identity")
        for name, identity in referenced.items():
            if identity != expected:
                raise SystemExit(f"{name}: model-target identity drifted from {expected}")


    def summary(registry):
        print(",".join(entry["name"] for entry in registry["profiles"]))


    modes = {
        "authorities": lambda registry: authorities(registry),
        "gates": lambda registry: gates(registry),
        "receipts": lambda registry: receipts(registry),
        "summary": lambda registry: summary(registry),
    }

    mode = sys.argv[1]
    registry = load(sys.argv[2])
    if mode == "model-targets":
        model_targets(registry, sys.argv[3])
    else:
        modes[mode](registry)
  '';
in
assert lib.assertMsg (data.schema_version == 1) "runtime profile registry schema version must be 1";
assert lib.assertMsg (
  lib.sort (a: b: a < b) names == expectedNames
) "runtime profile registry must carry exactly ${lib.concatStringsSep ", " expectedNames}";
# The served port is a client-facing contract value, not an incidental default.
assert lib.assertMsg (
  (profile "qwen38").port == 8000
) "the qwen38 serve profile must keep the served port 8000";
# The registry references the model-target identity instead of redefining it, and
# this pin is what keeps `model-targets/profiles.ncl` free of launch data.
assert lib.assertMsg (
  (profile "rwkv7-ttnn").model_target_blake3 == rwkvIdentity
) "the rwkv7-ttnn profile must reference the live RWKV model-target identity";
assert lib.assertMsg (
  (profile "metalium-rs-rwkv7").model_target_blake3 == rwkvIdentity
) "the metalium-rs-rwkv7 profile must reference the live RWKV model-target identity";
assert lib.assertMsg (
  modelTargetReferences != [ ]
) "at least one runtime profile must reference a model-target identity";
assert lib.assertMsg (modelTargetReferenceMismatch == [ ]) ''
  runtime profile model-target names must resolve to the identity each profile records: ${
    lib.concatMapStringsSep ", " (entry: entry.name) modelTargetReferenceMismatch
  }
'';
pkgs.runCommand "model-runtime-profile-registry-check"
  {
    nativeBuildInputs = [
      b3sum
      nickel
      python3
    ];
    inherit rwkvIdentity;
  }
  ''
        set -euo pipefail
        set -x
        cp -r ${repositorySource} "$PWD/repository"
        chmod -R u+w "$PWD/repository"
        cd "$PWD/repository"

        nickel typecheck model-targets/runtime-contract.ncl

        fresh_export="$PWD/runtime-profiles.json"
        nickel export --format json model-targets/runtime-profiles.ncl >"$fresh_export"
        cmp "$fresh_export" model-targets/generated/runtime-profiles.json

        # Each authority digest is the BLAKE3 of the document the profile names.
        python3 ${verify} authorities "$fresh_export" >"$PWD/authorities.txt"
        test -s "$PWD/authorities.txt"
        while read -r authority_path authority_blake3; do
          actual=$(b3sum "$authority_path" | cut -d' ' -f1)
          test "$actual" = "$authority_blake3"
        done <"$PWD/authorities.txt"

        # Every receipt the registry pins must exist, match its digest, and be admitted.
        python3 ${verify} receipts "$fresh_export" >"$PWD/receipts.txt"
        test -s "$PWD/receipts.txt"
        while read -r receipt_path receipt_blake3; do
          actual=$(b3sum "$receipt_path" | cut -d' ' -f1)
          test "$actual" = "$receipt_blake3"
        done <"$PWD/receipts.txt"

        # A gate artifact is package-relative, so it must not be absolute and must not
    # escape its package. The digest is bound to the installed program by
    # `tt-model check`, which holds the resolved package path.
    python3 ${verify} gates "$fresh_export" >"$PWD/gates.txt"
    while read -r gate_profile gate_artifact; do
      case "$gate_artifact" in
        /* | *..*)
          echo "gate artifact must be package-relative: $gate_profile $gate_artifact" >&2
          exit 1
          ;;
      esac
    done <"$PWD/gates.txt"

    # Every model-target reference must be the identity the target registry holds.
        python3 ${verify} model-targets "$fresh_export" "$rwkvIdentity"

    ${lib.concatMapStringsSep "\n" (entry: ''
      if nickel export --format json "model-targets/fixtures/${entry.file}" \
          >"$PWD/invalid.json" 2>"$PWD/invalid.log"; then
        echo "Nickel accepted invalid runtime profile fixture: ${entry.file}" >&2
        exit 1
      fi
      if [ -s "$PWD/invalid.json" ]; then
        echo "invalid runtime profile ${entry.file} emitted output despite failing" >&2
        exit 1
      fi
      grep -F ${lib.escapeShellArg entry.message} "$PWD/invalid.log"
    '') fixtures}

        mkdir -p "$out"
        cp "$fresh_export" "$out/runtime-profiles.json"
        {
          echo "schema_version=1"
          echo "profiles=$(python3 ${verify} summary "$fresh_export")"
          echo "authority_documents_verified=$(wc -l <"$PWD/authorities.txt")"
          echo "receipts_verified=$(wc -l <"$PWD/receipts.txt")"
          echo "fixtures_rejected=${toString (lib.length fixtures)}"
          echo "rwkv_model_target_identity=$rwkvIdentity"
          echo "export_blake3=$(b3sum "$fresh_export" | cut -d' ' -f1)"
        } >"$out/summary.txt"
  ''
