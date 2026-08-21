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
        hash = "sha256-9LTd41glc0HD4bCyfqoQMUdrPrvxTTxhTrF4VODKJE8=";
        chip = "blackhole";
        topology = "x1";
        hostSystem = "x86_64-linux";
        elfMachine = "Advanced Micro Devices X86-64";
      };
      x2 = mkAsset {
        name = "libttsim_bh_x2.so";
        hash = "sha256-wUI4Qy+QMCzg3fxdj1ymJ1b8OLXyLugr999EDZmbZE8=";
        chip = "blackhole";
        topology = "x2";
        hostSystem = "x86_64-linux";
        elfMachine = "Advanced Micro Devices X86-64";
      };
      x32 = mkAsset {
        name = "libttsim_bh_x32.so";
        hash = "sha256-7SfPqCiuVbn6FqOf/9aCsVjanUHyRvdoiiA/Wt1X+UM=";
        chip = "blackhole";
        topology = "x32";
        hostSystem = "x86_64-linux";
        elfMachine = "Advanced Micro Devices X86-64";
      };
    };
    wormhole = {
      x1 = mkAsset {
        name = "libttsim_wh.so";
        hash = "sha256-WQ9tOFH0/po56jU1PDXO7AfLNyVp+NlzvuThlSxHQZQ=";
        chip = "wormhole";
        topology = "x1";
        hostSystem = "x86_64-linux";
        elfMachine = "Advanced Micro Devices X86-64";
      };
      x2 = mkAsset {
        name = "libttsim_wh_x2.so";
        hash = "sha256-ilAO9L9BuqGcc5SNPcO9YrI/+i3v/GnKL0YEbL6btNo=";
        chip = "wormhole";
        topology = "x2";
        hostSystem = "x86_64-linux";
        elfMachine = "Advanced Micro Devices X86-64";
      };
      x8 = mkAsset {
        name = "libttsim_wh_x8.so";
        hash = "sha256-qWFCoG5yuKFs/s7UulfwFpp6Lj4i+yWyzocbL6DUXGQ=";
        chip = "wormhole";
        topology = "x8";
        hostSystem = "x86_64-linux";
        elfMachine = "Advanced Micro Devices X86-64";
      };
      x32 = mkAsset {
        name = "libttsim_wh_x32.so";
        hash = "sha256-/F/zj2eWmSiOnS8E5hr3mL3LlklYcVhCVwzFH/qhZcQ=";
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
        hash = "sha256-70k7MU8HW1pr3V2DMGhpEWZS5MfHbdf4k/DrhjFfskg=";
        chip = "blackhole";
        topology = "x1";
        hostSystem = "aarch64-linux";
        elfMachine = "AArch64";
      };
      x2 = mkAsset {
        name = "libttsim_bh_x2_aarch64.so";
        hash = "sha256-5YO/zk0B+ndaQcR5cIjClRcH242RBvtL7wnv1PZi2sg=";
        chip = "blackhole";
        topology = "x2";
        hostSystem = "aarch64-linux";
        elfMachine = "AArch64";
      };
    };
    wormhole = {
      x1 = mkAsset {
        name = "libttsim_wh_aarch64.so";
        hash = "sha256-K3ObudWuid0mghozYqGa9E3ksn3JiDo0cPL2OvUsbW4=";
        chip = "wormhole";
        topology = "x1";
        hostSystem = "aarch64-linux";
        elfMachine = "AArch64";
      };
      x2 = mkAsset {
        name = "libttsim_wh_x2_aarch64.so";
        hash = "sha256-UFF6UHhipUQfySUkWusox6Oz3IowPSQdbpYuf/X9lqg=";
        chip = "wormhole";
        topology = "x2";
        hostSystem = "aarch64-linux";
        elfMachine = "AArch64";
      };
      x8 = mkAsset {
        name = "libttsim_wh_x8_aarch64.so";
        hash = "sha256-Uh79SwMU8k7oFvDZ6R9u/FxdqbSccOvOs/4+DEbOaCU=";
        chip = "wormhole";
        topology = "x8";
        hostSystem = "aarch64-linux";
        elfMachine = "AArch64";
      };
      x32 = mkAsset {
        name = "libttsim_wh_x32_aarch64.so";
        hash = "sha256-uNxzQFgP9Gcsxuo2j1AnA8TNeKK5MrQuu16hklRUvZI=";
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
