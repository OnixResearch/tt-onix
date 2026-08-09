{
  lib,
  stdenv,
  cmake,
  ninja,
  pkg-config,
  makeWrapper,
  symlinkJoin,
  libblake3,
  nlohmann_json,
  enchantum,
  fmt,
  spdlog,
  xtensor,
  tt-logger,
  tt-metal,
  tt-umd,
  targetContext,
}:
let
  profile = targetContext.profile;
  artifacts = targetContext.artifacts;
  model = artifacts.checkpoint.path;
  artifactBlake3 = lib.mapAttrs (_role: artifact: artifact.blake3) artifacts;
  metaliumRuntime = symlinkJoin {
    name = "rwkv7-p150x2-metalium-runtime";
    paths = [ "${tt-metal}/libexec/tt-metalium" ];
    postBuild = ''
      unary_header_root=tt_metal/hw/inc/api/compute/eltwise_unary
      for header in ${tt-metal.src}/$unary_header_root/*.h; do
        install -Dm444 "$header" "$out/$unary_header_root/$(basename "$header")"
      done
      test -f "$out/$unary_header_root/lerp.h"
      test -f "$out/$unary_header_root/snake_beta.h"
    '';
  };
  metaliumRoot = "${metaliumRuntime}";
  meshDescriptor = "${metaliumRoot}/tt_metal/fabric/mesh_graph_descriptors/p150_x2_mesh_graph_descriptor.textproto";
  singleCardMeshDescriptor = "${metaliumRoot}/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto";
in
stdenv.mkDerivation {
  pname = "rwkv7-p150x2-runtime";
  version = "0.1.0";

  src = lib.cleanSource ./.;
  strictDeps = true;

  nativeBuildInputs = [
    cmake
    makeWrapper
    ninja
    pkg-config
  ];
  buildInputs = [
    enchantum
    fmt
    libblake3
    nlohmann_json
    spdlog
    tt-logger
    tt-metal
    tt-umd
    xtensor
  ];

  cmakeFlags = [
    (lib.cmakeFeature "RWKV7_CHECKPOINT_PATH" (toString model))
    (lib.cmakeFeature "RWKV7_MESH_DESCRIPTOR_PATH" meshDescriptor)
    (lib.cmakeFeature "RWKV7_SINGLE_CARD_MESH_DESCRIPTOR_PATH" singleCardMeshDescriptor)
    (lib.cmakeFeature "RWKV7_METALIUM_ROOT" metaliumRoot)
    (lib.cmakeFeature "RWKV7_TARGET_NAME" profile.target_name)
    (lib.cmakeFeature "RWKV7_MODEL_ID" profile.model.id)
    (lib.cmakeFeature "RWKV7_MODEL_REVISION" profile.model.revision)
    (lib.cmakeFeature "RWKV7_ARCHITECTURE_FAMILY" profile.model.architecture_family)
    (lib.cmakeFeature "RWKV7_STATE_KIND" profile.state.kind)
    (lib.cmakeFeature "RWKV7_MODEL_PRECISION" profile.model.precision)
    (lib.cmakeFeature "RWKV7_ARCHITECTURE_ADAPTER" profile.adapters.architecture)
    (lib.cmakeFeature "RWKV7_EVIDENCE_ADAPTER" profile.adapters.evidence)
    (lib.cmakeFeature "RWKV7_ARTIFACT_BLAKE3_JSON" (builtins.toJSON artifactBlake3))
    (lib.cmakeFeature "RWKV7_CHECKPOINT_BLAKE3" artifacts.checkpoint.blake3)
    (lib.cmakeFeature "RWKV7_TOPOLOGY_ID" profile.topology.id)
    (lib.cmakeFeature "RWKV7_REPLICATION" profile.topology.replication)
    (lib.cmakeFeature "RWKV7_CHECKPOINT_BYTES" (toString profile.resources.checkpoint_bytes))
    (lib.cmakeFeature "RWKV7_LAYER_COUNT" (toString profile.resources.layer_count))
    (lib.cmakeFeature "RWKV7_HIDDEN_SIZE" (toString profile.resources.hidden_size))
    (lib.cmakeFeature "RWKV7_VOCABULARY_SIZE" (toString profile.resources.vocabulary_size))
    (lib.cmakeFeature "RWKV7_MAXIMUM_PROMPT_TOKENS" (toString profile.resources.maximum_prompt_tokens))
  ];

  postInstall = ''
    makeWrapper "$out/libexec/rwkv7-p150x2/rwkv7-p150x2-runtime" \
      "$out/bin/rwkv7-p150x2-runtime" \
      --set TT_METAL_HOME ${lib.escapeShellArg metaliumRoot} \
      --set TT_METAL_RUNTIME_ROOT ${lib.escapeShellArg metaliumRoot} \
      --set TT_MESH_GRAPH_DESC_PATH ${lib.escapeShellArg meshDescriptor}
    mkdir -p "$out/share/model-target"
    echo '${profile.profile_identity}' >"$out/share/model-target/profile-blake3"
  '';

  doInstallCheck = true;
  installCheckPhase = ''
    runHook preInstallCheck
    "$out/bin/rwkv7-p150x2-runtime" --self-test >self-test.log
    grep -F 'rwkv7 P150x2 runtime self-test: PASS' self-test.log
    runHook postInstallCheck
  '';

  passthru = {
    inherit
      artifactBlake3
      meshDescriptor
      singleCardMeshDescriptor
      model
      profile
      ;
    profileIdentity = profile.profile_identity;
    targetName = profile.target_name;
  };

  meta = {
    description = "Pinned full-model RWKV-7 TT-NN runtime for the P150x2 mesh";
    license = lib.licenses.mit;
    mainProgram = "rwkv7-p150x2-runtime";
    platforms = [ "x86_64-linux" ];
  };
}
