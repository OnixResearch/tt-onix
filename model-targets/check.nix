{
  lib,
  pkgs,
  nickel,
  modelTargets,
}:
let
  rwkvProfile = modelTargets.getProfile "rwkv7-goose-world2.8-0.1b-p150x2";
  transformerProfile = modelTargets.getProfile "synthetic-transformer-kv-cache-fixture";
  rwkvAdapter = modelTargets.mkArchitectureAdapter rwkvProfile;
  transformerAdapter = modelTargets.mkArchitectureAdapter transformerProfile;
  rwkvEvidenceAdapter = modelTargets.mkEvidenceAdapter rwkvProfile;
  transformerEvidenceAdapter = modelTargets.mkEvidenceAdapter transformerProfile;
  rwkvTopology = modelTargets.mkTopology rwkvProfile;
  transformerTopology = modelTargets.mkTopology transformerProfile;

  fixtureArtifacts = profile:
    builtins.listToAttrs (
      map (artifact: {
        name = artifact.role;
        value = {
          path = "fixture:${artifact.role}";
          inherit (artifact) blake3 nix_hash;
        };
      }) profile.artifacts
    );
  rwkvArtifacts = fixtureArtifacts rwkvProfile;
  transformerArtifacts = fixtureArtifacts transformerProfile;

  dummyBuilders = {
    runtime = _context: "runtime";
    evidence = _context: "evidence";
    command = _context: "command";
    sourceBoundary = _context: "source-boundary";
    closure = _context: "closure";
  };

  mkRwkvFixture = overrides:
    modelTargets.mkModelTarget (
      {
        profile = rwkvProfile;
        artifacts = rwkvArtifacts;
        architectureAdapter = rwkvAdapter;
        evidenceAdapter = rwkvEvidenceAdapter;
        topology = rwkvTopology;
        builders = dummyBuilders;
      }
      // overrides
    );

  forceTarget = target:
    builtins.deepSeq [
      target.runtime
      target.command
      target.evidence
      target.sourceBoundary
      target.closure
      target.runnableOutputExposed
    ] true;

  positiveRwkv = builtins.tryEval (forceTarget (mkRwkvFixture { }));
  positiveTransformer = builtins.tryEval (
    forceTarget (
      modelTargets.mkModelTarget {
        profile = transformerProfile;
        artifacts = transformerArtifacts;
        architectureAdapter = transformerAdapter;
        evidenceAdapter = transformerEvidenceAdapter;
        topology = transformerTopology;
        builders = dummyBuilders;
      }
    )
  );
  architectureMismatch = builtins.tryEval (
    forceTarget (mkRwkvFixture {
      architectureAdapter = rwkvAdapter // { architectureFamilies = [ "other" ]; };
    })
  );
  stateMismatch = builtins.tryEval (
    forceTarget (mkRwkvFixture {
      architectureAdapter = rwkvAdapter // { stateKinds = [ "kv_cache" ]; };
    })
  );
  evidenceMismatch = builtins.tryEval (
    forceTarget (mkRwkvFixture {
      evidenceAdapter = rwkvEvidenceAdapter // { id = "wrong-evidence-adapter"; };
    })
  );
  artifactMismatch = builtins.tryEval (
    forceTarget (mkRwkvFixture {
      artifacts = rwkvArtifacts // {
        checkpoint = rwkvArtifacts.checkpoint // {
          blake3 = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
        };
      };
    })
  );
  precisionMismatch = builtins.tryEval (
    forceTarget (mkRwkvFixture {
      architectureAdapter = rwkvAdapter // { precisions = [ "fp32" ]; };
    })
  );
  topologyMismatch = builtins.tryEval (
    forceTarget (mkRwkvFixture {
      topology = rwkvTopology // { id = "other-topology"; };
    })
  );
in
assert lib.assertMsg positiveRwkv.success "compatible RWKV model target must evaluate";
assert lib.assertMsg positiveTransformer.success "compatible transformer fixture must evaluate";
assert lib.assertMsg (!architectureMismatch.success) "architecture mismatch must fail";
assert lib.assertMsg (!stateMismatch.success) "state mismatch must fail";
assert lib.assertMsg (!evidenceMismatch.success) "evidence mismatch must fail";
assert lib.assertMsg (!artifactMismatch.success) "artifact mismatch must fail";
assert lib.assertMsg (!precisionMismatch.success) "precision mismatch must fail";
assert lib.assertMsg (!topologyMismatch.success) "topology mismatch must fail";
pkgs.runCommand "model-target-profile-and-factory-check"
  {
    nativeBuildInputs = [ nickel ];
  }
  ''
    set -euo pipefail
    set -x
    model_targets_source=${./.} 
    cp -r "$model_targets_source" "$PWD/source"
    chmod -R u+w "$PWD/source"
    export_path="$PWD/out-profiles.json"
    (cd "$PWD/source" && nickel export --format json profiles.ncl >"$export_path")
    cmp "$export_path" ${./generated/profiles.json}

    for fixture in \
      invalid-missing-revision.ncl \
      invalid-state-kind.ncl \
      invalid-hash.ncl; do
      if (cd "$PWD/source" && nickel export --format json "fixtures/$fixture" \
          >"$PWD/invalid.json" 2>"$PWD/invalid.log"); then
        echo "Nickel accepted invalid model profile fixture: $fixture" >&2
        exit 1
      fi
      if [ -s "$PWD/invalid.json" ]; then
        echo "Invalid profile $fixture emitted output despite failing" >&2
        exit 1
      fi
    done

    if cmp ${./fixtures/drifted-profiles.json} ${./generated/profiles.json}; then
      echo "generated model profile drift fixture matched canonical data" >&2
      exit 1
    fi

    grep -F 'rwkv7-goose-world2.8-0.1b-p150x2' "$export_path"
    grep -F 'synthetic-transformer-kv-cache-fixture' "$export_path"
    grep -F '"runnable": false' "$export_path"
    touch "$out"
  ''
