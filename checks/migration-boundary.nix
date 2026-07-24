{
  lib,
  runCommand,
}:
let
  requiredPaths = [
    "docs/migration-provenance.md"
    "pkgs/rwkv-lab"
    "pkgs/rwkv-layer-harness"
    "pkgs/rwkv-ttwkv7-boundary-device"
    "pkgs/rwkv-ttwkv7-persistent-device"
    "pkgs/rwkv-ttwkv7-persistent-evidence"
    "pkgs/rwkv-ttwkv7-persistent-passed-evidence"
    "pkgs/tt-vibethinker-bench"
    "pkgs/ttsim"
    "pkgs/ttwkv7"
    "pkgs/ttwkv7-owner-control"
  ];
  forbiddenPaths = [
    "inventory"
    "machines"
    "sops"
    "vars"
  ];
  classifyBoundary =
    entries:
    let
      missing = builtins.filter (path: !(builtins.elem path entries)) requiredPaths;
      forbidden = builtins.filter (path: builtins.elem path entries) forbiddenPaths;
    in
    {
      inherit forbidden missing;
      valid = missing == [ ] && forbidden == [ ];
    };
  positiveFixture = classifyBoundary requiredPaths;
  missingFixture = classifyBoundary (builtins.tail requiredPaths);
  forbiddenFixture = classifyBoundary (requiredPaths ++ [ (builtins.head forbiddenPaths) ]);
  source = lib.cleanSource ../.;
in
assert positiveFixture.valid;
assert !missingFixture.valid;
assert missingFixture.missing != [ ];
assert !forbiddenFixture.valid;
assert forbiddenFixture.forbidden != [ ];
runCommand "tenstorrent-migration-boundary" { } ''
  set -euo pipefail

  source_root=${lib.escapeShellArg source}

  for required_path in ${lib.escapeShellArgs requiredPaths}; do
    test -e "$source_root/$required_path"
  done

  for forbidden_path in ${lib.escapeShellArgs forbiddenPaths}; do
    test ! -e "$source_root/$forbidden_path"
  done

  mkdir -p "$out"
  cp "$source_root/docs/migration-provenance.md" "$out/provenance.md"
''
