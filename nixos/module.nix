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
  qwen38 = cfg.qwen38;

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
  defaultQwenListenAddress = "127.0.0.1";
  defaultQwenPort = 8000;
  defaultQwenMaximumSequenceLength = 2048;
  defaultQwenMaximumGenerationTokens = 64;
  qwenRestartSeconds = 10;
  qwenStopTimeoutSeconds = 120;
  qwenStateDirectory = "qwen38-p150x2";
  qwenStateDirectoryMode = "0750";

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

    qwen38 = {
      enable = mkEnableOption "the serialized Qwen3.8-27B P150x2 inference service";

      package = mkOption {
        type = types.package;
        default = pkgs.qwen36;
        defaultText = lib.literalExpression "pkgs.qwen36";
        description = "Qwen package that owns the P150x2 service executable.";
      };

      modelPath = mkOption {
        type = types.str;
        default = "";
        example = "/var/lib/qwen38-p150x2/model";
        description = "Absolute path to the pinned Qwen3.8-27B snapshot and P300 tensor cache.";
      };

      modelAlias = mkOption {
        type = types.str;
        default = "Qwen3.8-27B";
        description = "Model identifier exposed by the OpenAI-compatible API.";
      };

      listenAddress = mkOption {
        type = types.str;
        default = defaultQwenListenAddress;
        description = "Address for the serialized HTTP service.";
      };

      port = mkOption {
        type = types.port;
        default = defaultQwenPort;
        description = "TCP port for the serialized HTTP service.";
      };

      maximumSequenceLength = mkOption {
        type = types.ints.positive;
        default = defaultQwenMaximumSequenceLength;
        description = "Maximum prompt and generation token count.";
      };

      maximumGenerationTokens = mkOption {
        type = types.ints.positive;
        default = defaultQwenMaximumGenerationTokens;
        description = "Maximum generated token count for one request.";
      };

      conflictingUnits = mkOption {
        type = types.listOf types.str;
        default = [ ];
        example = [
          "llamacpp-server.service"
          "docker-tt-inference-server.service"
        ];
        description = "Inference units that must stop before Qwen owns both P150 devices.";
      };
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

    (mkIf qwen38.enable {
      assertions = [
        {
          assertion = lib.hasPrefix "/" qwen38.modelPath;
          message = "hardware.tenstorrent.qwen38.modelPath must be an absolute path";
        }
        {
          assertion = qwen38.maximumGenerationTokens <= qwen38.maximumSequenceLength;
          message = "Qwen maximumGenerationTokens must not exceed maximumSequenceLength";
        }
      ];

      systemd.services.qwen38-p150x2 = {
        description = "Qwen3.8-27B serialized OpenAI-compatible service on P150x2";
        after = [ "network-online.target" ];
        wants = [ "network-online.target" ];
        wantedBy = [ "multi-user.target" ];
        conflicts = qwen38.conflictingUnits;

        environment = {
          HF_HUB_OFFLINE = "1";
          HF_MODEL = qwen38.modelPath;
          HOME = "/var/lib/${qwenStateDirectory}";
          MESH_DEVICE = "P150x2";
          MPLCONFIGDIR = "/var/lib/${qwenStateDirectory}/matplotlib";
          PYTHONUNBUFFERED = "1";
          QWEN35_TEMP = "0";
          TT_CACHE_PATH = qwen38.modelPath;
          XDG_CACHE_HOME = "/var/lib/${qwenStateDirectory}/cache";
        };

        unitConfig.ConditionPathExists = [
          qwen38.modelPath
          "/dev/tenstorrent/0"
          "/dev/tenstorrent/1"
        ];

        serviceConfig = {
          ExecStart = lib.escapeShellArgs [
            "${qwen38.package}/bin/qwen36-p150x2-serve"
            "--host"
            qwen38.listenAddress
            "--port"
            (toString qwen38.port)
            "--model-path"
            qwen38.modelPath
            "--model-alias"
            qwen38.modelAlias
            "--max-sequence-length"
            (toString qwen38.maximumSequenceLength)
            "--max-generation-tokens"
            (toString qwen38.maximumGenerationTokens)
          ];
          Group = cfg.deviceGroup;
          Restart = "on-failure";
          RestartSec = "${toString qwenRestartSeconds}s";
          StateDirectory = qwenStateDirectory;
          StateDirectoryMode = qwenStateDirectoryMode;
          TimeoutStartSec = 0;
          TimeoutStopSec = qwenStopTimeoutSeconds;
          UnsetEnvironment = [
            "TT_MESH_GRAPH_DESC_PATH"
            "TT_METAL_SIMULATOR"
          ];
          User = "root";
          WorkingDirectory = "/var/lib/${qwenStateDirectory}";
        };
      };
    })

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
