{
  lib,
  stdenv,
  runCommand,
  fetchurl,
  binutils,
  bash,
  coreutils,
  gnugrep,
  jq,
  tt-metal,
  ttsim,
}:
let
  authority = import ./assets.nix { inherit lib; };
  x86Blackhole = authority.selectAsset {
    system = "x86_64-linux";
    chip = "blackhole";
    topology = "x1";
  };
  armBlackhole = authority.selectAsset {
    system = "aarch64-linux";
    chip = "blackhole";
    topology = "x1";
  };
  unsupportedSystem = builtins.tryEval (
    (authority.selectAsset {
      system = "riscv64-linux";
      chip = "blackhole";
      topology = "x1";
    }).name
  );
  unsupportedChip = builtins.tryEval (
    (authority.selectAsset {
      system = "x86_64-linux";
      chip = "quasar";
      topology = "x1";
    }).name
  );
  missingAsset = builtins.tryEval (
    (authority.selectAsset {
      system = "aarch64-linux";
      chip = "blackhole";
      topology = "x32";
    }).name
  );
  localBlackhole = authority.selectAsset {
    system = stdenv.hostPlatform.system;
    chip = "blackhole";
    topology = "x1";
  };
  oppositeBlackhole =
    if stdenv.hostPlatform.system == "x86_64-linux" then armBlackhole else x86Blackhole;
  oppositeHostAsset = fetchurl {
    url = "https://github.com/tenstorrent/ttsim/releases/download/v${ttsim.version}/${oppositeBlackhole.name}";
    hash = oppositeBlackhole.hash;
  };
in
# Reject a package bump that leaves emitted receipts on an older version.
assert lib.hasInfix ''pub const SUPPORTED_SIMULATOR_VERSION: &str = "${ttsim.version}";'' (
  builtins.readFile ../ttsim-runtime-validator/src/lib.rs
);
assert x86Blackhole.name == "libttsim_bh.so";
assert armBlackhole.name == "libttsim_bh_aarch64.so";
assert x86Blackhole.hash != armBlackhole.hash;
assert unsupportedSystem.success == false;
assert unsupportedChip.success == false;
assert missingAsset.success == false;
runCommand "ttsim-runtime-bundle-check"
  {
    nativeBuildInputs = [
      binutils
      bash
      coreutils
      gnugrep
      jq
    ];
  }
  ''
    set -euo pipefail

    required_symbols=(
      libttsim_init
      libttsim_exit
      libttsim_set_pci_dma_mem_callbacks
      libttsim_pci_config_rd32
      libttsim_pci_config_wr32
      libttsim_pci_mem_rd_bytes
      libttsim_pci_mem_wr_bytes
      libttsim_clock
      libttsim_tile_rd_bytes
      libttsim_tile_wr_bytes
    )

    validate_layout() {
      runtime_dir="$1"
      expected_descriptor="$2"
      expected_machine="$3"
      test -L "$runtime_dir/libttsim.so" || {
        echo 'ttsim layout: canonical library is missing' >&2
        return 1
      }
      test -f "$runtime_dir/soc_descriptor.yaml" || {
        echo 'ttsim layout: SoC descriptor is missing' >&2
        return 1
      }
      cmp "$runtime_dir/soc_descriptor.yaml" "$expected_descriptor" >/dev/null || {
        echo 'ttsim layout: SoC descriptor does not match the chip' >&2
        return 1
      }
      readelf -h "$runtime_dir/libttsim.so" \
        | grep -F 'Machine:' \
        | grep -F "$expected_machine" >/dev/null || {
        echo 'ttsim layout: simulator asset does not match the host architecture' >&2
        return 1
      }
      for symbol in "''${required_symbols[@]}"; do
        nm -D --defined-only "$runtime_dir/libttsim.so" \
          | grep -E "[[:space:]]$symbol$" >/dev/null || {
          echo "ttsim layout: required ABI symbol is missing: $symbol" >&2
          return 1
        }
      done
    }

    blackhole_runtime=${ttsim}/lib/ttsim/runtimes/blackhole-x1
    wormhole_runtime=${ttsim}/lib/ttsim/runtimes/wormhole-x1
    blackhole_descriptor=${tt-metal}/libexec/tt-metalium/tt_metal/soc_descriptors/blackhole_140_arch.yaml
    wormhole_descriptor=${tt-metal}/libexec/tt-metalium/tt_metal/soc_descriptors/wormhole_b0_80_arch.yaml
    validate_layout "$blackhole_runtime" "$blackhole_descriptor" '${localBlackhole.elfMachine}'
    validate_layout "$wormhole_runtime" "$wormhole_descriptor" '${localBlackhole.elfMachine}'

    cp -r --no-preserve=mode "$blackhole_runtime" "$TMPDIR/missing-descriptor"
    rm "$TMPDIR/missing-descriptor/soc_descriptor.yaml"
    if validate_layout "$TMPDIR/missing-descriptor" "$blackhole_descriptor" '${localBlackhole.elfMachine}' \
      >"$TMPDIR/missing-descriptor.log" 2>&1; then
      echo 'missing-descriptor fixture passed runtime validation' >&2
      exit 1
    fi
    grep -F 'ttsim layout: SoC descriptor is missing' "$TMPDIR/missing-descriptor.log"

    cp -r --no-preserve=mode "$blackhole_runtime" "$TMPDIR/wrong-descriptor"
    cp "$wormhole_descriptor" "$TMPDIR/wrong-descriptor/soc_descriptor.yaml"
    if validate_layout "$TMPDIR/wrong-descriptor" "$blackhole_descriptor" '${localBlackhole.elfMachine}' \
      >"$TMPDIR/wrong-descriptor.log" 2>&1; then
      echo 'wrong-descriptor fixture passed runtime validation' >&2
      exit 1
    fi
    grep -F 'ttsim layout: SoC descriptor does not match the chip' "$TMPDIR/wrong-descriptor.log"

    mkdir -p "$TMPDIR/missing-symbol"
    ln -s ${coreutils}/bin/true "$TMPDIR/missing-symbol/libttsim.so"
    cp "$blackhole_descriptor" "$TMPDIR/missing-symbol/soc_descriptor.yaml"
    if validate_layout "$TMPDIR/missing-symbol" "$blackhole_descriptor" '${localBlackhole.elfMachine}' \
      >"$TMPDIR/missing-symbol.log" 2>&1; then
      echo 'missing-symbol fixture passed runtime validation' >&2
      exit 1
    fi
    grep -F 'ttsim layout: required ABI symbol is missing: libttsim_init' "$TMPDIR/missing-symbol.log"

    if readelf -h ${oppositeHostAsset} \
      | grep -F 'Machine:' \
      | grep -F '${localBlackhole.elfMachine}' >/dev/null; then
      echo 'opposite-host fixture matched the local host architecture' >&2
      exit 1
    fi
    readelf -h ${oppositeHostAsset} | grep -F 'Machine:' | grep -F '${oppositeBlackhole.elfMachine}'

    TT_MESH_GRAPH_DESC_PATH=/wrong-mesh \
      ${coreutils}/bin/env -u TT_METAL_DISABLE_SFPLOADMACRO \
      ${ttsim}/bin/ttsim-blackhole ${bash}/bin/bash -c \
      'printf "%s\n%s\n%s\n%s\n" "$TT_METAL_SIMULATOR" "$TT_METAL_SLOW_DISPATCH_MODE" "''${TT_MESH_GRAPH_DESC_PATH-unset}" "''${TT_METAL_DISABLE_SFPLOADMACRO-unset}"' \
      > "$TMPDIR/wrapper-environment"
    grep -Fx '${ttsim}/lib/ttsim/runtimes/blackhole-x1/libttsim.so' "$TMPDIR/wrapper-environment"
    test "$(grep -Fxc '1' "$TMPDIR/wrapper-environment")" -eq 1
    test "$(grep -Fxc 'unset' "$TMPDIR/wrapper-environment")" -eq 2

    ${ttsim}/bin/ttsim-blackhole-sfpu ${bash}/bin/bash -c \
      'printf "%s\n" "$TT_METAL_DISABLE_SFPLOADMACRO"' \
      > "$TMPDIR/sfpu-wrapper-environment"
    grep -Fx '1' "$TMPDIR/sfpu-wrapper-environment"

    jq -e '.schemaVersion == 1' ${ttsim}/share/ttsim/runtime-manifest.json >/dev/null
    jq -e '.unsupported.quasar | contains("no package-matched Quasar")' \
      ${ttsim}/share/ttsim/runtime-manifest.json >/dev/null
    jq -e '.unsupported.multiChip | contains("packages no reviewed blackhole_P300 descriptor")' \
      ${ttsim}/share/ttsim/runtime-manifest.json >/dev/null

    if ${ttsim}/bin/ttsim-blackhole-x2 >"$TMPDIR/blackhole-x2.log" 2>&1; then
      echo 'blocked blackhole-x2 launcher started successfully' >&2
      exit 1
    fi
    grep -F 'missing_package_matched_multi_chip_authority' "$TMPDIR/blackhole-x2.log"
    grep -F 'packages no reviewed blackhole_P300 cluster descriptor' "$TMPDIR/blackhole-x2.log"
    jq -e '.simulated == false and .status == "blocked"' \
      ${ttsim}/share/ttsim/blackhole-x2-blocker.json >/dev/null

    touch "$out"
  ''
