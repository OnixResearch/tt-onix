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
        hash = "sha256-kzOyLIoa7vxPqoOs71bXnIpDx73cT/KfOWXUo3Ys9iY=";
        chip = "blackhole";
        topology = "x1";
        hostSystem = "x86_64-linux";
        elfMachine = "Advanced Micro Devices X86-64";
      };
      x2 = mkAsset {
        name = "libttsim_bh_x2.so";
        hash = "sha256-CZVJZDHHlpPuKHKaQPuUE0b8xQp1B3xlvz6ECRu8SeY=";
        chip = "blackhole";
        topology = "x2";
        hostSystem = "x86_64-linux";
        elfMachine = "Advanced Micro Devices X86-64";
      };
      x32 = mkAsset {
        name = "libttsim_bh_x32.so";
        hash = "sha256-DL6RxwcE0hF4VtfKRkM+h+Wak1xVAUcDMiZGbuY+Ky0=";
        chip = "blackhole";
        topology = "x32";
        hostSystem = "x86_64-linux";
        elfMachine = "Advanced Micro Devices X86-64";
      };
    };
    wormhole = {
      x1 = mkAsset {
        name = "libttsim_wh.so";
        hash = "sha256-JobrwhLql1PArs3nTmQK/0PvvlTbAyJnZifDoMGxTYE=";
        chip = "wormhole";
        topology = "x1";
        hostSystem = "x86_64-linux";
        elfMachine = "Advanced Micro Devices X86-64";
      };
      x2 = mkAsset {
        name = "libttsim_wh_x2.so";
        hash = "sha256-TI26YUX2B62H7udBVB+Sw3v8JpKe71AzAFoz8cXR8GM=";
        chip = "wormhole";
        topology = "x2";
        hostSystem = "x86_64-linux";
        elfMachine = "Advanced Micro Devices X86-64";
      };
      x8 = mkAsset {
        name = "libttsim_wh_x8.so";
        hash = "sha256-jOvOlrbFgh4pjRUAJ6gRmhv5KS+KWv4+WJo8Ppv5CGg=";
        chip = "wormhole";
        topology = "x8";
        hostSystem = "x86_64-linux";
        elfMachine = "Advanced Micro Devices X86-64";
      };
      x32 = mkAsset {
        name = "libttsim_wh_x32.so";
        hash = "sha256-EtbyPfD/lcoUdZiaSpKbbJh3OoPNFuUZDJKs/dIAkW4=";
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
        hash = "sha256-kMjaNKatlFXEJp8JhQ4mBYUxX5vvSHoyJZEtDCXDIdg=";
        chip = "blackhole";
        topology = "x1";
        hostSystem = "aarch64-linux";
        elfMachine = "AArch64";
      };
      x2 = mkAsset {
        name = "libttsim_bh_x2_aarch64.so";
        hash = "sha256-Skpb6P0zFNpSyj8UEMaRB5gepTMvde7waNFgL7tRuW0=";
        chip = "blackhole";
        topology = "x2";
        hostSystem = "aarch64-linux";
        elfMachine = "AArch64";
      };
    };
    wormhole = {
      x1 = mkAsset {
        name = "libttsim_wh_aarch64.so";
        hash = "sha256-04FCydlFJqXWtI5cBxHv1xj/rtknAVDFVW2g3niCFK4=";
        chip = "wormhole";
        topology = "x1";
        hostSystem = "aarch64-linux";
        elfMachine = "AArch64";
      };
      x2 = mkAsset {
        name = "libttsim_wh_x2_aarch64.so";
        hash = "sha256-uxugo6y0GXMRcdZQiMOXHsdp/7/jxhcNSWUAm3hcmNA=";
        chip = "wormhole";
        topology = "x2";
        hostSystem = "aarch64-linux";
        elfMachine = "AArch64";
      };
      x8 = mkAsset {
        name = "libttsim_wh_x8_aarch64.so";
        hash = "sha256-5s6IjmPwzd1NIFGqgv9sPjz7aX1GM8GTDvEoT9yFpPs=";
        chip = "wormhole";
        topology = "x8";
        hostSystem = "aarch64-linux";
        elfMachine = "AArch64";
      };
      x32 = mkAsset {
        name = "libttsim_wh_x32_aarch64.so";
        hash = "sha256-xMAROppECkdv6DigkYKUvoGYKbUHrHs181jdZw3fx7I=";
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
    multiChip = "package-matched TT-Metal has mock-cluster plumbing but packages no reviewed blackhole_P300 descriptor; ttsim 1.10.6 documents incomplete multi-chip UMD support and validates Blackhole on micah/native-sim";
  };
}
