{
  lib,
  rustPlatform,
  nickel,
  jq,
  coreutils,
  ttsim,
  tt-smi,
  targetContext,
  deviceRuntime,
}:
let
  profile = targetContext.profile;
  artifacts = targetContext.artifacts;
  artifactBlake3 = lib.mapAttrs (_role: artifact: artifact.blake3) artifacts;
  packageSource = lib.fileset.toSource {
    root = ../.;
    fileset = lib.fileset.unions [
      ./.
      ../model-target-core
    ];
  };
in
rustPlatform.buildRustPackage {
  pname = "rwkv7-p150x2-evidence";
  version = "0.1.0";

  src = packageSource;
  sourceRoot = "source/rwkv7-p150x2-evidence";
  cargoLock.lockFile = ./Cargo.lock;
  nativeBuildInputs = [ nickel ];

  MODEL_TARGET_PROFILE_BLAKE3 = profile.profile_identity;
  MODEL_TARGET_NAME = profile.target_name;
  MODEL_TARGET_MODEL_ID = profile.model.id;
  MODEL_TARGET_REVISION = profile.model.revision;
  MODEL_TARGET_ARCHITECTURE_FAMILY = profile.model.architecture_family;
  MODEL_TARGET_STATE_KIND = profile.state.kind;
  MODEL_TARGET_PRECISION = profile.model.precision;
  MODEL_TARGET_ARCHITECTURE_ADAPTER = profile.adapters.architecture;
  MODEL_TARGET_EVIDENCE_ADAPTER = profile.adapters.evidence;
  MODEL_TARGET_ARTIFACT_BLAKE3_JSON = builtins.toJSON artifactBlake3;
  MODEL_TARGET_TOPOLOGY_ID = profile.topology.id;
  MODEL_TARGET_MESH_SHAPE_JSON = builtins.toJSON profile.topology.mesh_shape;
  MODEL_TARGET_PHYSICAL_DEVICE_IDS_JSON = builtins.toJSON profile.topology.physical_device_ids;
  MODEL_TARGET_REPLICATION = profile.topology.replication;
  MODEL_TARGET_REQUIRED_OPERATIONS_JSON = builtins.toJSON targetContext.architectureAdapter.requiredOperations;
  MODEL_TARGET_CHECKPOINT_BLAKE3 = artifacts.checkpoint.blake3;
  MODEL_TARGET_TOKENIZER_VOCABULARY_BLAKE3 = artifacts.tokenizer_vocabulary.blake3;
  RWKV7_TOKENIZER_VOCABULARY = artifacts.tokenizer_vocabulary.path;
  RWKV7_DEVICE_RUNTIME = deviceRuntime;
  TT_SMI_PATH = tt-smi;

  postInstall = ''
    mkdir -p "$out/share/model-target" "$out/share/rwkv7-p150x2-throughput"
    echo '${profile.profile_identity}' >"$out/share/model-target/profile-blake3"
    cp throughput-contracts.ncl throughput-profile.ncl \
      fixtures/throughput-workload.json \
      "$out/share/rwkv7-p150x2-throughput/"
    nickel export --format json throughput-profile.ncl \
      >"$out/share/rwkv7-p150x2-throughput/throughput-profile.json"
    substitute run-simulator-smoke.sh \
      "$out/bin/rwkv7-p150x2-simulator-smoke" \
      --replace-fail '@ttsim@' ${lib.escapeShellArg (toString ttsim)} \
      --replace-fail '@timeout@' ${lib.escapeShellArg "${coreutils}/bin/timeout"} \
      --replace-fail '@runtime@' ${lib.escapeShellArg (toString deviceRuntime)}
    chmod +x "$out/bin/rwkv7-p150x2-simulator-smoke"
  '';

  doCheck = true;
  doInstallCheck = true;
  nativeInstallCheckInputs = [
    nickel
    jq
  ];
  installCheckPhase = ''
    runHook preInstallCheck
    "$out/bin/rwkv7-p150x2-evidence" --self-test
    "$out/bin/rwkv7-p150x2-command" --self-test
    bash -n "$out/bin/rwkv7-p150x2-simulator-smoke"
    if "$out/bin/rwkv7-p150x2-throughput-benchmark" \
      >missing-benchmark-args.out 2>missing-benchmark-args.err; then
      echo "throughput benchmark accepted missing arguments" >&2
      exit 1
    fi
    grep -F 'usage:' missing-benchmark-args.err

    # r[verify tenstorrent.rwkv7_p150x2.throughput_profile.valid]
    nickel export --format json throughput-profile.ncl >throughput-profile-first.json
    nickel export --format json throughput-profile.ncl >throughput-profile-second.json
    cmp throughput-profile-first.json throughput-profile-second.json
    if nickel export --format json fixtures/invalid-throughput-lockstep.ncl \
      >invalid-profile.json 2>invalid-profile.log; then
      echo "lockstep throughput profile passed its typed contract" >&2
      exit 1
    fi
    test -s invalid-profile.log

    throughput_core="$out/bin/rwkv7-p150x2-throughput"
    profile_identity="$($throughput_core profile-identity throughput-profile-first.json)"
    jq --arg identity "$profile_identity" \
      '.profile_identity_blake3 = $identity' \
      fixtures/throughput-valid.json >throughput-raw.json

    # r[verify tenstorrent.rwkv7_p150x2.throughput_admission.valid]
    "$throughput_core" summarize throughput-profile-first.json throughput-raw.json \
      >throughput-receipt.json
    "$throughput_core" admit throughput-profile-first.json throughput-raw.json \
      throughput-receipt.json >throughput-admission.json
    grep -F '"status": "admitted"' throughput-admission.json

    # r[verify tenstorrent.rwkv7_p150x2.throughput_admission.estimated]
    jq '.two_card_runs[0].physical_hardware = false' throughput-raw.json \
      >throughput-invalid-estimated.json
    if "$throughput_core" summarize throughput-profile-first.json \
      throughput-invalid-estimated.json >invalid-estimated.out 2>invalid-estimated.err; then
      echo "estimated throughput evidence passed admission" >&2
      exit 1
    fi
    grep -F 'unattributed_throughput_run' invalid-estimated.out
    runHook postInstallCheck
  '';

  passthru = {
    inherit deviceRuntime profile;
    targetIdentity = {
      profileBlake3 = profile.profile_identity;
      targetName = profile.target_name;
      modelId = profile.model.id;
      revision = profile.model.revision;
      artifactBlake3 = artifactBlake3;
    };
  };

  meta = {
    description = "Fail-closed raw evidence classifier for RWKV-7 generation on P150x2";
    license = lib.licenses.mit;
    platforms = [ "x86_64-linux" ];
    mainProgram = "rwkv7-p150x2-evidence";
  };
}
