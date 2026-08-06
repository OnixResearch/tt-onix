{
  config,
  pkgs,
  lib,
  ...
}:
let
  inherit (lib)
    mkEnableOption
    mkIf
    mkMerge
    mkOption
    types
    ;

  cfg = config.hardware.tenstorrent;
  training = cfg.training;

  defaultDriverPackage = config.boot.kernelPackages.callPackage ../pkgs/tt-kmd { };
  admittedTrainingDriverVersion = "2.10.0";
  admittedTrainingMesh = "p150_x2";
  admittedTrainingHugepageMount = "/dev/hugepages-1G";
  admittedTrainingEvidenceParent = "/var/lib/tenstorrent-training";
  admittedTrainingEvidenceRoot = "${admittedTrainingEvidenceParent}/evidence";
  defaultTrainingHugepageCount = 8;
  defaultDeviceMode = "0660";
  permissiveDeviceMode = "0666";
  defaultHugepageMode = "0770";
  permissiveHugepageMode = "0777";
  evidenceParentMode = "0750";
  evidenceRootMode = "0770";

  deviceMode =
    if cfg.accessPolicy == "group-restricted" then defaultDeviceMode else permissiveDeviceMode;
  hugepageMode =
    if cfg.accessPolicy == "group-restricted" then defaultHugepageMode else permissiveHugepageMode;

  meshDescriptorFile =
    if lib.hasSuffix ".textproto" cfg.meshName then
      cfg.meshName
    else
      "${cfg.meshName}_mesh_graph_descriptor.textproto";
  meshDescriptorPath = "${cfg.descriptorPackage}/libexec/tt-metalium/tt_metal/fabric/mesh_graph_descriptors/${meshDescriptorFile}";
in
{
  disabledModules = [ "hardware/tenstorrent.nix" ];

  options.hardware.tenstorrent = {
    enable = mkEnableOption "Tenstorrent driver and utilities";

    driverPackage = mkOption {
      type = types.package;
      default = defaultDriverPackage;
      defaultText = lib.literalExpression "config.boot.kernelPackages.tt-kmd";
      description = "Kernel-matched TT-KMD package installed by this module.";
    };

    descriptorPackage = mkOption {
      type = types.package;
      default = pkgs.tt-metal;
      defaultText = lib.literalExpression "pkgs.tt-metal";
      description = "TT-Metal package that owns the selected mesh descriptor.";
    };

    deviceGroup = mkOption {
      type = types.str;
      default = "tenstorrent";
      description = "Group that can access Tenstorrent devices and hugepages.";
    };

    accessPolicy = mkOption {
      type = types.enum [
        "group-restricted"
        "permissive"
      ];
      default = "group-restricted";
      description = ''
        Device and hugepage access policy. The group-restricted policy uses
        mode 0660 for devices and 0770 for hugepages. The permissive policy
        is an explicit compatibility option.
      '';
    };

    meshName = mkOption {
      type = types.nullOr types.str;
      default = null;
      example = "p150_x4";
      description = ''
        Mesh graph descriptor naming this machine's Tenstorrent topology.
        Give a descriptor stem or a complete textproto filename.
      '';
    };

    training = {
      enable = mkEnableOption "the admitted dual-P150A training host profile";

      requiredDriverVersion = mkOption {
        type = types.str;
        default = admittedTrainingDriverVersion;
        readOnly = true;
        description = "Exact TT-KMD version required by training admission.";
      };

      hugepageMount = mkOption {
        type = types.str;
        default = admittedTrainingHugepageMount;
        description = "Named 1 GiB hugepage mount used by training containers.";
      };

      hugepageCount = mkOption {
        type = types.ints.positive;
        default = defaultTrainingHugepageCount;
        description = "Number of 1 GiB hugepages reserved at boot.";
      };

      evidenceRoot = mkOption {
        type = types.str;
        default = admittedTrainingEvidenceRoot;
        readOnly = true;
        description = "Package-authorized parent for fresh training evidence runs.";
      };
    };
  };

  config = mkIf cfg.enable (mkMerge [
    {
      nixpkgs.overlays = [ (import ../overlay.nix) ];

      boot = {
        extraModulePackages = [ cfg.driverPackage ];
        kernelModules = [ "tenstorrent" ];
      };

      services.udev = {
        packages = [ cfg.driverPackage ];
        extraRules = ''
          SUBSYSTEM=="tenstorrent", GROUP="${cfg.deviceGroup}", MODE="${deviceMode}"
        '';
      };

      users.groups.${cfg.deviceGroup} = { };

      environment.systemPackages = with pkgs; [
        tt-smi
        tt-system-tools
      ];

      environment.variables = mkIf (cfg.meshName != null) {
        TT_MESH_GRAPH_DESC_PATH = meshDescriptorPath;
      };
    }

    (mkIf training.enable {
      assertions = [
        {
          assertion = lib.getVersion cfg.driverPackage == training.requiredDriverVersion;
          message = ''
            hardware.tenstorrent.driverPackage version ${lib.getVersion cfg.driverPackage}
            does not match admitted training driver ${training.requiredDriverVersion}
          '';
        }
        {
          assertion = cfg.meshName == admittedTrainingMesh;
          message = "the admitted training profile requires meshName = ${admittedTrainingMesh}";
        }
        {
          assertion = cfg.descriptorPackage.outPath == pkgs.tt-metal.outPath;
          message = "the admitted training profile requires the package-owned TT-Metal descriptor";
        }
        {
          assertion = training.hugepageMount == admittedTrainingHugepageMount;
          message = "the admitted training profile requires hugepageMount = ${admittedTrainingHugepageMount}";
        }
      ];

      boot.kernelParams = [
        "hugepagesz=1G"
        "hugepages=${toString training.hugepageCount}"
      ];

      systemd.mounts = [
        {
          what = "hugetlbfs";
          where = training.hugepageMount;
          type = "hugetlbfs";
          options = "pagesize=1024M,mode=${hugepageMode},gid=${cfg.deviceGroup}";
          wantedBy = [ "sysinit.target" ];
        }
      ];

      systemd.tmpfiles.rules = [
        "d ${admittedTrainingEvidenceParent} ${evidenceParentMode} root ${cfg.deviceGroup} - -"
        "d ${training.evidenceRoot} ${evidenceRootMode} root ${cfg.deviceGroup} - -"
      ];

      virtualisation.docker.enable = true;
    })
  ]);
}
