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
  nickel,
  enchantum,
  fmt,
  spdlog,
  xtensor,
  tt-logger,
  tt-metal,
  tt-umd,
  targetContext,
  fetchQueueDiagnostics ? false,
  fusedWkvBf16ReadoutBoundary ? false,
  persistentDecodeProfile ? ./persistent-decode-profile.ncl,
  maximumGeneratedTokens ? 8,
  runtimeName ? null,
}:
let
  profile = targetContext.profile;
  artifacts = targetContext.artifacts;
  model = artifacts.checkpoint.path;
  resolvedRuntimeName =
    if runtimeName != null then
      runtimeName
    else if fetchQueueDiagnostics then
      "rwkv7-p150x2-fetch-queue-diagnostic-runtime"
    else
      "rwkv7-p150x2-runtime";
  artifactBlake3 = lib.mapAttrs (_role: artifact: artifact.blake3) artifacts;
  persistentDecodeProfileFile = builtins.baseNameOf (toString persistentDecodeProfile);
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
  pname = resolvedRuntimeName;
  version = "0.1.0";

  src = lib.cleanSource ./.;
  strictDeps = true;

  nativeBuildInputs = [
    cmake
    makeWrapper
    ninja
    nickel
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

  preConfigure = ''
    nickel export --format json ${lib.escapeShellArg persistentDecodeProfileFile} \
      >persistent-decode-profile.json
  '';

  cmakeFlags = [
    (lib.cmakeFeature "RWKV7_CHECKPOINT_PATH" (toString model))
    (lib.cmakeFeature "RWKV7_PERSISTENT_DECODE_PROFILE_PATH" "${placeholder "out"}/share/rwkv7-p150x2/persistent-decode-profile.json")
    (lib.cmakeFeature "RWKV7_MAXIMUM_GENERATED_TOKENS" (toString maximumGeneratedTokens))
    (lib.cmakeFeature "RWKV7_RUNTIME_NAME" resolvedRuntimeName)
    (lib.cmakeFeature "RWKV7_MESH_DESCRIPTOR_PATH" meshDescriptor)
    (lib.cmakeFeature "RWKV7_SINGLE_CARD_MESH_DESCRIPTOR_PATH" singleCardMeshDescriptor)
    (lib.cmakeFeature "RWKV7_METALIUM_ROOT" metaliumRoot)
    (lib.cmakeFeature "RWKV7_TARGET_NAME" profile.target_name)
    (lib.cmakeFeature "RWKV7_TARGET_PROFILE_BLAKE3" profile.profile_identity)
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
    (lib.cmakeBool "RWKV7_FETCH_QUEUE_DIAGNOSTICS" fetchQueueDiagnostics)
    (lib.cmakeBool "RWKV7_FUSED_WKV_BF16_READOUT_BOUNDARY" fusedWkvBf16ReadoutBoundary)
  ];

  postInstall = ''
    makeWrapper "$out/libexec/rwkv7-p150x2/rwkv7-p150x2-runtime" \
      "$out/bin/rwkv7-p150x2-runtime" \
      --set TT_METAL_HOME ${lib.escapeShellArg metaliumRoot} \
      --set TT_METAL_RUNTIME_ROOT ${lib.escapeShellArg metaliumRoot} \
      --set TT_MESH_GRAPH_DESC_PATH ${lib.escapeShellArg meshDescriptor} \
      --set RWKV7_FUSED_WKV_KERNEL_ROOT "$out/share/rwkv7-p150x2/kernel"
    install -Dm444 ../persistent-decode-profile.json \
      "$out/share/rwkv7-p150x2/persistent-decode-profile.json"
    mkdir -p "$out/share/model-target"
    echo '${profile.profile_identity}' >"$out/share/model-target/profile-blake3"
  '';

  doInstallCheck = true;
  installCheckPhase = ''
    runHook preInstallCheck
    nickel export --format json ../${lib.escapeShellArg persistentDecodeProfileFile} \
      >persistent-decode-profile-check.json
    cmp persistent-decode-profile-check.json \
      "$out/share/rwkv7-p150x2/persistent-decode-profile.json"
    if nickel export --format json ../fixtures/invalid-persistent-decode-profile.ncl; then
      echo 'invalid persistent decode profile passed validation' >&2
      exit 1
    fi
    nickel export --format json ../persistent-decode-window8-candidate-profile.ncl \
      >persistent-decode-window8-candidate-profile.json
    if nickel export --format json \
      ../fixtures/invalid-persistent-decode-window8-profile.ncl; then
      echo 'invalid window-8 candidate profile passed validation' >&2
      exit 1
    fi
    nickel export --format json ../persistent-decode-window16-candidate-profile.ncl \
      >persistent-decode-window16-candidate-profile.json
    if nickel export --format json \
      ../fixtures/invalid-persistent-decode-window16-profile.ncl; then
      echo 'invalid window-16 candidate profile passed validation' >&2
      exit 1
    fi
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
