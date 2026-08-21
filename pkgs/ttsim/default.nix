{
  lib,
  stdenv,
  fetchurl,
  autoPatchelfHook,
  makeWrapper,
  binutils,
  coreutils,
  b3sum,
  jq,
  gcc-unwrapped,
  tt-metal,
}:
let
  # r[impl tenstorrent.ttsim.runtime_bundle]
  version = "1.10.2";
  releaseBaseUrl = "https://github.com/tenstorrent/ttsim/releases/download/v${version}";
  assetAuthority = import ./assets.nix { inherit lib; };
  hostSystem = stdenv.hostPlatform.system;
  selectedAssets = assetAuthority.flattenSystemAssets hostSystem;
  fetchedAssets = map (
    asset:
    asset
    // {
      source = fetchurl {
        url = "${releaseBaseUrl}/${asset.name}";
        inherit (asset) hash;
      };
    }
  ) selectedAssets;

  descriptors = {
    blackhole = "${tt-metal}/libexec/tt-metalium/tt_metal/soc_descriptors/blackhole_140_arch.yaml";
    wormhole = "${tt-metal}/libexec/tt-metalium/tt_metal/soc_descriptors/wormhole_b0_80_arch.yaml";
  };

  runtimeAssets = map (
    chip:
    assetAuthority.selectAsset {
      system = hostSystem;
      inherit chip;
      topology = "x1";
    }
  ) (lib.attrNames assetAuthority.supportedRuntimeTopologies);

  installAssets = lib.concatMapStringsSep "\n" (asset: ''
    install -Dm755 ${asset.source} "$out/lib/ttsim/assets/${asset.name}"
    ln -s "ttsim/assets/${asset.name}" "$out/lib/${asset.name}"
  '') fetchedAssets;

  installRuntimes = lib.concatMapStringsSep "\n" (asset: ''
    runtime_dir="$out/lib/ttsim/runtimes/${asset.chip}-${asset.topology}"
    mkdir -p "$runtime_dir"
    cp "$out/lib/ttsim/assets/${asset.name}" "$runtime_dir/${asset.name}"
    ln -s "${asset.name}" "$runtime_dir/libttsim.so"
    cp ${descriptors.${asset.chip}} "$runtime_dir/soc_descriptor.yaml"
  '') runtimeAssets;

  requiredAbiSymbols = [
    "libttsim_init"
    "libttsim_exit"
    "libttsim_set_pci_dma_mem_callbacks"
    "libttsim_pci_config_rd32"
    "libttsim_pci_config_wr32"
    "libttsim_pci_mem_rd_bytes"
    "libttsim_pci_mem_wr_bytes"
    "libttsim_clock"
    "libttsim_tile_rd_bytes"
    "libttsim_tile_wr_bytes"
  ];

  validateAssets = lib.concatMapStringsSep "\n" (asset: ''
    readelf -h "$out/lib/ttsim/assets/${asset.name}" \
      | grep -F 'Machine:' \
      | grep -F '${asset.elfMachine}' >/dev/null
    for symbol in ${lib.escapeShellArgs requiredAbiSymbols}; do
      nm -D --defined-only "$out/lib/ttsim/assets/${asset.name}" \
        | grep -E "[[:space:]]$symbol$" >/dev/null
    done
  '') fetchedAssets;

  validateRuntimes = lib.concatMapStringsSep "\n" (asset: ''
    runtime_dir="$out/lib/ttsim/runtimes/${asset.chip}-${asset.topology}"
    test -L "$runtime_dir/libttsim.so"
    test -f "$runtime_dir/soc_descriptor.yaml"
    cmp "$runtime_dir/soc_descriptor.yaml" ${descriptors.${asset.chip}}
  '') runtimeAssets;

  runtimeManifest = builtins.toJSON {
    schemaVersion = 1;
    simulatorVersion = version;
    ttMetalVersion = tt-metal.version;
    inherit hostSystem requiredAbiSymbols;
    assets = map (asset: {
      inherit (asset)
        chip
        elfMachine
        hash
        hostSystem
        name
        topology
        ;
    }) selectedAssets;
    runtimes = map (asset: {
      inherit (asset) chip topology;
      assetName = asset.name;
      descriptorSource = descriptors.${asset.chip};
      canonicalLibrary = "lib/ttsim/runtimes/${asset.chip}-${asset.topology}/libttsim.so";
      descriptor = "lib/ttsim/runtimes/${asset.chip}-${asset.topology}/soc_descriptor.yaml";
      dispatchMode = "slow";
    }) runtimeAssets;
    unsupported = assetAuthority.unsupportedReleaseSurfaces;
  };
in
stdenv.mkDerivation (finalAttrs: {
  pname = "ttsim";
  inherit version;

  dontUnpack = true;
  dontStrip = true;

  nativeBuildInputs = [
    autoPatchelfHook
    b3sum
    binutils
    jq
    makeWrapper
  ];

  buildInputs = [
    gcc-unwrapped.lib
    stdenv.cc.cc.lib
  ];

  installPhase = ''
    runHook preInstall

    mkdir -p "$out/bin" "$out/lib/ttsim/assets" "$out/share/ttsim"
    ${installAssets}
    ${installRuntimes}

    cat > "$out/share/ttsim/runtime-manifest.json" <<'EOF'
    ${runtimeManifest}
    EOF

    makeWrapper ${coreutils}/bin/env "$out/bin/ttsim-blackhole" \
      --unset TT_MESH_GRAPH_DESC_PATH \
      --set TT_METAL_SIMULATOR "$out/lib/ttsim/runtimes/blackhole-x1/libttsim.so" \
      --set TT_METAL_SLOW_DISPATCH_MODE 1
    makeWrapper ${coreutils}/bin/env "$out/bin/ttsim-wormhole" \
      --unset TT_MESH_GRAPH_DESC_PATH \
      --set TT_METAL_SIMULATOR "$out/lib/ttsim/runtimes/wormhole-x1/libttsim.so" \
      --set TT_METAL_SLOW_DISPATCH_MODE 1
    makeWrapper ${coreutils}/bin/env "$out/bin/ttsim-blackhole-sfpu" \
      --unset TT_MESH_GRAPH_DESC_PATH \
      --set TT_METAL_SIMULATOR "$out/lib/ttsim/runtimes/blackhole-x1/libttsim.so" \
      --set TT_METAL_SLOW_DISPATCH_MODE 1 \
      --set TT_METAL_DISABLE_SFPLOADMACRO 1
    makeWrapper ${coreutils}/bin/env "$out/bin/ttsim-wormhole-sfpu" \
      --unset TT_MESH_GRAPH_DESC_PATH \
      --set TT_METAL_SIMULATOR "$out/lib/ttsim/runtimes/wormhole-x1/libttsim.so" \
      --set TT_METAL_SLOW_DISPATCH_MODE 1 \
      --set TT_METAL_DISABLE_SFPLOADMACRO 1

    # r[impl tenstorrent.ttsim.multi_chip_boundary]
    bh_x2_library="$out/lib/ttsim/assets/${
      (assetAuthority.selectAsset {
        system = hostSystem;
        chip = "blackhole";
        topology = "x2";
      }).name
    }"
    bh_x2_blake3="$(b3sum "$bh_x2_library" | cut -d ' ' -f 1)"
    tt_metal_identity='tt-metal-${tt-metal.version}:${tt-metal}'
    tt_metal_blake3="$(printf '%s' "$tt_metal_identity" | b3sum | cut -d ' ' -f 1)"
    jq -n \
      --arg host_system '${hostSystem}' \
      --arg library "$bh_x2_library" \
      --arg library_blake3 "$bh_x2_blake3" \
      --arg tt_metal "$tt_metal_identity" \
      --arg tt_metal_blake3 "$tt_metal_blake3" \
      '{
        schema_version: 1,
        simulated: false,
        status: "blocked",
        topology: "blackhole-x2",
        blocker: {
          code: "missing_package_matched_multi_chip_authority",
          detail: "tt-metal ${tt-metal.version} packages no reviewed blackhole_P300 cluster descriptor; ttsim ${version} requires the unmerged mcraighead/mc-p300 integration"
        },
        identities: {
          host_system: $host_system,
          library: { value: $library, blake3: $library_blake3 },
          descriptor: { value: "blackhole_P300_both_mmio.yaml", status: "missing" },
          tt_metal: { value: $tt_metal, blake3: $tt_metal_blake3 }
        },
        claims: [],
        non_claims: [
          "physical placement",
          "hardware profiling",
          "performance",
          "thermals",
          "device health",
          "silicon success"
        ]
      }' > "$out/share/ttsim/blackhole-x2-blocker.json"

    cat > "$out/bin/ttsim-blackhole-x2" <<EOF
    #!${stdenv.shell}
    ${coreutils}/bin/cat "$out/share/ttsim/blackhole-x2-blocker.json"
    exit 1
    EOF
    chmod +x "$out/bin/ttsim-blackhole-x2"

    runHook postInstall
  '';

  postInstall = ''
    ${validateAssets}
    ${validateRuntimes}
  '';

  passthru = {
    inherit assetAuthority requiredAbiSymbols;
    manifest = runtimeManifest;
    runtimes = {
      blackhole = "${finalAttrs.finalPackage}/lib/ttsim/runtimes/blackhole-x1";
      wormhole = "${finalAttrs.finalPackage}/lib/ttsim/runtimes/wormhole-x1";
    };
  };

  meta = {
    description = "Complete, host-matched TT-Metal simulator runtime bundles";
    homepage = "https://github.com/tenstorrent/ttsim";
    license = lib.licenses.asl20;
    platforms = assetAuthority.supportedHostSystems;
    sourceProvenance = with lib.sourceTypes; [ binaryNativeCode ];
  };
})
