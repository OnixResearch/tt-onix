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
        hash = "sha256-snal+6JgZOty/pZBJuNLrUu/UVF1iQVwui6u9nD+1XI=";
        chip = "blackhole";
        topology = "x1";
        hostSystem = "x86_64-linux";
        elfMachine = "Advanced Micro Devices X86-64";
      };
      x2 = mkAsset {
        name = "libttsim_bh_x2.so";
        hash = "sha256-Ivz1uiNM1GEY5IPwLkKu9p2Pe2qUemcHSzjYiYO6LIo=";
        chip = "blackhole";
        topology = "x2";
        hostSystem = "x86_64-linux";
        elfMachine = "Advanced Micro Devices X86-64";
      };
      x32 = mkAsset {
        name = "libttsim_bh_x32.so";
        hash = "sha256-zYyqia5+gqePg3wAuIbwD3q+140IBYeLRXwv3UHBpo4=";
        chip = "blackhole";
        topology = "x32";
        hostSystem = "x86_64-linux";
        elfMachine = "Advanced Micro Devices X86-64";
      };
    };
    wormhole = {
      x1 = mkAsset {
        name = "libttsim_wh.so";
        hash = "sha256-TwklPMcHx6g2J7ELpITvNgP97UgvEQ/Wn8yf7C6k2iM=";
        chip = "wormhole";
        topology = "x1";
        hostSystem = "x86_64-linux";
        elfMachine = "Advanced Micro Devices X86-64";
      };
      x2 = mkAsset {
        name = "libttsim_wh_x2.so";
        hash = "sha256-/VyrOkuLwkSFEMhwI6o9wwa44t4iXNqXLIdIAHEJz1E=";
        chip = "wormhole";
        topology = "x2";
        hostSystem = "x86_64-linux";
        elfMachine = "Advanced Micro Devices X86-64";
      };
      x8 = mkAsset {
        name = "libttsim_wh_x8.so";
        hash = "sha256-lS6vBsOStwPFwsaKmLmku9kCs9r1eG7ROSF5l0GYUMw=";
        chip = "wormhole";
        topology = "x8";
        hostSystem = "x86_64-linux";
        elfMachine = "Advanced Micro Devices X86-64";
      };
      x32 = mkAsset {
        name = "libttsim_wh_x32.so";
        hash = "sha256-hZGdqET3CFgrEnxDqrClcIqqQnQCHiMdICafJ5SxYLk=";
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
        hash = "sha256-LfsXU1t8+9/uPnYNwCOi16P3uVyGM/YAm4/hrSzRIE4=";
        chip = "blackhole";
        topology = "x1";
        hostSystem = "aarch64-linux";
        elfMachine = "AArch64";
      };
      x2 = mkAsset {
        name = "libttsim_bh_x2_aarch64.so";
        hash = "sha256-OLFIs1zaZB+7K866+dvBo4+g9vJJ6IRTAoxmM2Fto/4=";
        chip = "blackhole";
        topology = "x2";
        hostSystem = "aarch64-linux";
        elfMachine = "AArch64";
      };
    };
    wormhole = {
      x1 = mkAsset {
        name = "libttsim_wh_aarch64.so";
        hash = "sha256-pEKfe9UJ9QarAFMj43NsF1oZztr2y5ESh8kkJnkkmio=";
        chip = "wormhole";
        topology = "x1";
        hostSystem = "aarch64-linux";
        elfMachine = "AArch64";
      };
      x2 = mkAsset {
        name = "libttsim_wh_x2_aarch64.so";
        hash = "sha256-S40GeSpPcBgW4oBTyx2Z5aDSlXq3qrTpMQqjdKjII5U=";
        chip = "wormhole";
        topology = "x2";
        hostSystem = "aarch64-linux";
        elfMachine = "AArch64";
      };
      x8 = mkAsset {
        name = "libttsim_wh_x8_aarch64.so";
        hash = "sha256-6w5UwDFahhGK2/wlO92DSh2N+VuEj1W5atgtke8j5rw=";
        chip = "wormhole";
        topology = "x8";
        hostSystem = "aarch64-linux";
        elfMachine = "AArch64";
      };
      x32 = mkAsset {
        name = "libttsim_wh_x32_aarch64.so";
        hash = "sha256-qvKtDFk8DCq16P411An2WMDO81uVWOrCumBx8tg0KLQ=";
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
