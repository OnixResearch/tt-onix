{
  lib,
  stdenv,
  cmake,
  ninja,
  pkg-config,
  makeWrapper,
  libblake3,
  nlohmann_json,
  enchantum,
  fmt,
  spdlog,
  xtensor,
  tt-logger,
  tt-metal,
  tt-umd,
  model,
}:
let
  metaliumRoot = "${tt-metal}/libexec/tt-metalium";
  meshDescriptor = "${metaliumRoot}/tt_metal/fabric/mesh_graph_descriptors/p150_x2_mesh_graph_descriptor.textproto";
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
    (lib.cmakeFeature "RWKV7_METALIUM_ROOT" metaliumRoot)
  ];

  postInstall = ''
    makeWrapper "$out/libexec/rwkv7-p150x2/rwkv7-p150x2-runtime" \
      "$out/bin/rwkv7-p150x2-runtime" \
      --set TT_METAL_HOME ${lib.escapeShellArg metaliumRoot} \
      --set TT_METAL_RUNTIME_ROOT ${lib.escapeShellArg metaliumRoot} \
      --set TT_MESH_GRAPH_DESC_PATH ${lib.escapeShellArg meshDescriptor}
  '';

  doInstallCheck = true;
  installCheckPhase = ''
    runHook preInstallCheck
    "$out/bin/rwkv7-p150x2-runtime" --self-test >self-test.log
    grep -F 'rwkv7 P150x2 runtime self-test: PASS' self-test.log
    runHook postInstallCheck
  '';

  passthru = {
    inherit meshDescriptor model;
  };

  meta = {
    description = "Pinned full-model RWKV-7 TT-NN runtime for the P150x2 mesh";
    license = lib.licenses.mit;
    mainProgram = "rwkv7-p150x2-runtime";
    platforms = [ "x86_64-linux" ];
  };
}
