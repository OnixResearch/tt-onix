{
  lib,
  rustPlatform,
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

  postInstall = ''
    mkdir -p "$out/share/model-target"
    echo '${profile.profile_identity}' >"$out/share/model-target/profile-blake3"
  '';

  doCheck = true;
  doInstallCheck = true;
  installCheckPhase = ''
    runHook preInstallCheck
    "$out/bin/rwkv7-p150x2-evidence" --self-test
    "$out/bin/rwkv7-p150x2-command" --self-test
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
