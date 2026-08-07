{ lib }:
# r[impl tenstorrent.ttsim.asset_selection]
let
  mkAsset =
    {
      name,
      hash,
      chip,
      topology,
      hostSystem,
      elfMachine,
    }:
    {
      inherit
        chip
        elfMachine
        hash
        hostSystem
        name
        topology
        ;
    };

  x86_64 = {
    blackhole = {
      x1 = mkAsset {
        name = "libttsim_bh.so";
        hash = "sha256-Y8kLMIHrirZAJCBCABYNxDp8OHj7T1Dn/wNzIZnp49U=";
        chip = "blackhole";
        topology = "x1";
        hostSystem = "x86_64-linux";
        elfMachine = "Advanced Micro Devices X86-64";
      };
      x2 = mkAsset {
        name = "libttsim_bh_x2.so";
        hash = "sha256-4Q3wpCCazxQ1oIjOlFaDuZg2iqvZxNug6YXCexY4OeQ=";
        chip = "blackhole";
        topology = "x2";
        hostSystem = "x86_64-linux";
        elfMachine = "Advanced Micro Devices X86-64";
      };
      x32 = mkAsset {
        name = "libttsim_bh_x32.so";
        hash = "sha256-USk4GpCf1Zy95jn0KEDkZ08DyEvWe05Mm781xIjBaxM=";
        chip = "blackhole";
        topology = "x32";
        hostSystem = "x86_64-linux";
        elfMachine = "Advanced Micro Devices X86-64";
      };
    };
    wormhole = {
      x1 = mkAsset {
        name = "libttsim_wh.so";
        hash = "sha256-Qgh3wkwdD+cpEJbzW0ORRAOuUjwWReph+VAuXHluiPA=";
        chip = "wormhole";
        topology = "x1";
        hostSystem = "x86_64-linux";
        elfMachine = "Advanced Micro Devices X86-64";
      };
      x2 = mkAsset {
        name = "libttsim_wh_x2.so";
        hash = "sha256-/nRil/amdwXs9qpS2083b0bsTorkqRbWg+7en45sjwc=";
        chip = "wormhole";
        topology = "x2";
        hostSystem = "x86_64-linux";
        elfMachine = "Advanced Micro Devices X86-64";
      };
      x8 = mkAsset {
        name = "libttsim_wh_x8.so";
        hash = "sha256-C6D+9iW9UUIuTcz4rpwsz2c0XbuG9cBg1Uw1e5+e3ec=";
        chip = "wormhole";
        topology = "x8";
        hostSystem = "x86_64-linux";
        elfMachine = "Advanced Micro Devices X86-64";
      };
      x32 = mkAsset {
        name = "libttsim_wh_x32.so";
        hash = "sha256-fydp5CWv9yDm8U8+es6t2nQR22gjgeaRhpaevSEpNI0=";
        chip = "wormhole";
        topology = "x32";
        hostSystem = "x86_64-linux";
        elfMachine = "Advanced Micro Devices X86-64";
      };
    };
  };

  aarch64 = {
    blackhole = {
      x1 = mkAsset {
        name = "libttsim_bh_aarch64.so";
        hash = "sha256-DGaC0AcDbYocohCYRQq6li4ekGs4vaoa4gJHtHt58bA=";
        chip = "blackhole";
        topology = "x1";
        hostSystem = "aarch64-linux";
        elfMachine = "AArch64";
      };
      x2 = mkAsset {
        name = "libttsim_bh_x2_aarch64.so";
        hash = "sha256-zdXeygka5PW1wNgsTYQtToG4rewneZQz8wbSPOA6sNM=";
        chip = "blackhole";
        topology = "x2";
        hostSystem = "aarch64-linux";
        elfMachine = "AArch64";
      };
    };
    wormhole = {
      x1 = mkAsset {
        name = "libttsim_wh_aarch64.so";
        hash = "sha256-797qt5AqCHc+W9Y5daNtdxPE5184TiIyjyINSCr7KSE=";
        chip = "wormhole";
        topology = "x1";
        hostSystem = "aarch64-linux";
        elfMachine = "AArch64";
      };
      x2 = mkAsset {
        name = "libttsim_wh_x2_aarch64.so";
        hash = "sha256-HP2RPpfkVRVXIC/fgonrCLNq8Xp7H9jQHveSW1WxUTU=";
        chip = "wormhole";
        topology = "x2";
        hostSystem = "aarch64-linux";
        elfMachine = "AArch64";
      };
      x8 = mkAsset {
        name = "libttsim_wh_x8_aarch64.so";
        hash = "sha256-vhe7qZVKzdQgqDSp4av7b7A6pvfBPSmgTAIME9G4Kto=";
        chip = "wormhole";
        topology = "x8";
        hostSystem = "aarch64-linux";
        elfMachine = "AArch64";
      };
      x32 = mkAsset {
        name = "libttsim_wh_x32_aarch64.so";
        hash = "sha256-Xxkhfh3S99zN4BZi5Cdt9vHb0nr9kkX30CaWLEmwV6U=";
        chip = "wormhole";
        topology = "x32";
        hostSystem = "aarch64-linux";
        elfMachine = "AArch64";
      };
    };
  };

  assetTable = {
    x86_64-linux = x86_64;
    aarch64-linux = aarch64;
  };

  selectAsset =
    {
      system,
      chip,
      topology,
    }:
    let
      systemAssets = assetTable.${system} or (throw "ttsim: unsupported host system `${system}`");
      chipAssets = systemAssets.${chip} or (throw "ttsim: unsupported chip `${chip}` on `${system}`");
    in
    chipAssets.${topology}
      or (throw "ttsim: unsupported topology `${topology}` for `${chip}` on `${system}`");

  flattenSystemAssets =
    system:
    let
      systemAssets = assetTable.${system} or (throw "ttsim: unsupported host system `${system}`");
    in
    lib.concatMap lib.attrValues (lib.attrValues systemAssets);
in
{
  inherit assetTable flattenSystemAssets selectAsset;

  supportedHostSystems = lib.attrNames assetTable;
  supportedRuntimeTopologies = {
    blackhole = [ "x1" ];
    wormhole = [ "x1" ];
  };

  unsupportedReleaseSurfaces = {
    quasar = "tt-metal 0.74.0 has no package-matched Quasar SoC descriptor";
    multiChip = "package-matched TT-Metal has mock-cluster plumbing but packages no reviewed blackhole_P300 descriptor; ttsim multi-chip needs the unmerged mcraighead/mc-p300 integration";
  };
}
