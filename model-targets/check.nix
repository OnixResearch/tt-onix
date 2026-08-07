{
  lib,
  pkgs,
  nickel,
  modelTargetConsumerGenerator,
  modelTargets,
}:
let
  repositorySource = lib.cleanSource ../.;
  rwkvProfile = modelTargets.getProfile "rwkv7-goose-world2.8-0.1b-p150x2";
  transformerProfile = modelTargets.getProfile "synthetic-transformer-kv-cache-fixture";
  rwkvAdapter = modelTargets.mkArchitectureAdapter rwkvProfile;
  transformerAdapter = modelTargets.mkArchitectureAdapter transformerProfile;
  rwkvEvidenceAdapter = modelTargets.mkEvidenceAdapter rwkvProfile;
  transformerEvidenceAdapter = modelTargets.mkEvidenceAdapter transformerProfile;
  rwkvTopology = modelTargets.mkTopology rwkvProfile;
  transformerTopology = modelTargets.mkTopology transformerProfile;

  fixtureArtifacts =
    profile:
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

  mkRwkvFixture =
    overrides:
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

  forceTarget =
    target:
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
      architectureAdapter = rwkvAdapter // {
        architectureFamilies = [ "other" ];
      };
    })
  );
  stateMismatch = builtins.tryEval (
    forceTarget (mkRwkvFixture {
      architectureAdapter = rwkvAdapter // {
        stateKinds = [ "kv_cache" ];
      };
    })
  );
  evidenceMismatch = builtins.tryEval (
    forceTarget (mkRwkvFixture {
      evidenceAdapter = rwkvEvidenceAdapter // {
        id = "wrong-evidence-adapter";
      };
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
      architectureAdapter = rwkvAdapter // {
        precisions = [ "fp32" ];
      };
    })
  );
  topologyMismatch = builtins.tryEval (
    forceTarget (mkRwkvFixture {
      topology = rwkvTopology // {
        id = "other-topology";
      };
    })
  );
  profileIdentityMismatch = builtins.tryEval (
    forceTarget (mkRwkvFixture {
      profile = rwkvProfile // {
        profile_identity = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
      };
    })
  );
  profileDimensionMismatch = builtins.tryEval (
    forceTarget (mkRwkvFixture {
      profile = rwkvProfile // {
        resources = rwkvProfile.resources // {
          hidden_size = rwkvProfile.resources.hidden_size + 1;
        };
      };
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
assert lib.assertMsg (!profileIdentityMismatch.success) "profile identity mismatch must fail";
assert lib.assertMsg (!profileDimensionMismatch.success) "profile dimension mismatch must fail";
pkgs.runCommand "model-target-profile-and-factory-check"
  {
    nativeBuildInputs = [
      modelTargetConsumerGenerator
      nickel
    ];
  }
  ''
    set -euo pipefail
    set -x
    cp -r ${repositorySource} "$PWD/repository"
    chmod -R u+w "$PWD/repository"
    cd "$PWD/repository"

    export_path="$PWD/out-profiles.json"
    nickel export --format json model-targets/profiles.ncl >"$export_path"
    cmp "$export_path" model-targets/generated/profiles.json
    model-target-consumer-generator check "$export_path" .

    for fixture in \
      invalid-missing-revision.ncl \
      invalid-state-kind.ncl \
      invalid-hash.ncl \
      invalid-hash-uppercase.ncl \
      invalid-hash-mixed-case.ncl \
      invalid-hash-short.ncl \
      invalid-hash-long.ncl \
      invalid-hash-nonhex.ncl \
      invalid-missing-consumer.ncl; do
      if nickel export --format json "model-targets/fixtures/$fixture" \
          >"$PWD/invalid.json" 2>"$PWD/invalid.log"; then
        echo "Nickel accepted invalid model profile fixture: $fixture" >&2
        exit 1
      fi
      if [ -s "$PWD/invalid.json" ]; then
        echo "Invalid profile $fixture emitted output despite failing" >&2
        exit 1
      fi
      grep -E 'revision|state|blake3|consumer' "$PWD/invalid.log"
    done

    nickel export --format json model-targets/fixtures/invalid-inconsistent-dimensions.ncl \
      >"$PWD/inconsistent.json"
    if model-target-consumer-generator check "$PWD/inconsistent.json" . \
        >"$PWD/inconsistent.log" 2>&1; then
      echo "consumer generator accepted inconsistent dimensions" >&2
      exit 1
    fi
    grep -F 'head_count * head_size' "$PWD/inconsistent.log"

    cp model-targets/generated/consumer-identities.rs "$PWD/fresh-rust.rs"
    printf '\n// stale fixture\n' >>model-targets/generated/consumer-identities.rs
    if model-target-consumer-generator check "$export_path" . \
        >"$PWD/stale.log" 2>&1; then
      echo "consumer generator accepted stale Rust output" >&2
      exit 1
    fi
    grep -F 'stale generated consumer projection' "$PWD/stale.log"
    grep -F 'model-targets/generated/consumer-identities.rs' "$PWD/stale.log"
    mv "$PWD/fresh-rust.rs" model-targets/generated/consumer-identities.rs
    model-target-consumer-generator check "$export_path" .

    cmp model-targets/generated/consumer-identities.rs \
      pkgs/rwkv-layer-harness/src/generated_model_target_identity.rs
    cmp model-targets/generated/consumer-identities.rs \
      pkgs/rwkv7-p150x2-evidence/src/generated_model_target_identity.rs
    cmp model-targets/generated/consumer_identities.py \
      pkgs/rwkv7-lora-reference/generated_model_target_identity.py
    cmp model-targets/generated/consumer-identities.h \
      pkgs/ttwkv7/generated-model-target-identity.h

    profile_identity=${rwkvProfile.profile_identity}
    for projection in \
      model-targets/generated/consumer-identities.rs \
      model-targets/generated/consumer_identities.py \
      model-targets/generated/consumer-identities.h \
      model-targets/generated/consumer-identities.nix; do
      grep -F "$profile_identity" "$projection"
    done

    if cmp model-targets/fixtures/drifted-profiles.json model-targets/generated/profiles.json; then
      echo "generated model profile drift fixture matched canonical data" >&2
      exit 1
    fi

    grep -F 'rwkv7-goose-world2.8-0.1b-p150x2' "$export_path"
    grep -F 'synthetic-transformer-kv-cache-fixture' "$export_path"
    grep -F '"runnable": false' "$export_path"
    touch "$out"
  ''
