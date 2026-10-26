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
  defaultQwenMaximumSequenceLength = 131072;
  defaultQwenMaximumGenerationTokens = 8192;
  qwenRestartSeconds = 10;
  qwenStopTimeoutSeconds = 120;
  qwenStateDirectory = "qwen38-p150x2";
  qwenStateDirectoryMode = "0750";

  deviceMode =
    if cfg.accessPolicy == "group-restricted" then defaultDeviceMode else permissiveDeviceMode;
  hugepageMode =
    if cfg.accessPolicy == "group-restricted" then defaultHugepageMode else permissiveDeviceMode;
  # The continuous-slot profiles are packaged beside each other; the option
  # names the one the service selects.
  continuousSlotPolicyFile =
    "continuous-slot-policy" + (if qwen38.continuousSlotProfile == "b8" then "-b8" else "") + ".json";

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
        default = pkgs.qwen38;
        defaultText = lib.literalExpression "pkgs.qwen38";
        description = "Qwen package that owns the P150x2 service executable.";
      };

      mtpSpeculation = mkOption {
        type = types.bool;
        default = false;
        description = ''
          Enable experimental greedy MTP serving on the serialized route.
          This option requires owner review of matched physical evidence.
          It does not admit sampled speculation or batch speculation.
        '';
      };

      sampledFusion = mkOption {
        type = types.bool;
        default = false;
        description = ''
          Serve sampled requests (temperature above zero) through the
          fused GDN decode path under the distributional admission gate.
          Deterministic requests always keep the exact-parity path. The
          sampled kernel fits 4 users per step on the shared runtime, so a
          batch of sampled requests is capped at 4 and the rest wait.
          Without this option the service accepts only temperature 0.
        '';
      };

      gdnDecodeStep = mkOption {
        type = types.bool;
        default = true;
        description = ''
          Serve decode through the fused GDN decode step: one conv-prep and one
          recurrence program per GDN layer instead of the composite op chain.
          Greedy requests stay on this route; sampled requests keep the
          sampled-fusion route when that is enabled. The step changes greedy
          tokens relative to the composite route under a numerical quality gate,
          not bit-exact parity.
        '';
      };

      continuousSlotScheduler = mkOption {
        type = types.bool;
        default = false;
        description = ''
          Serve generation through the continuous-slot scheduler instead of the
          group coordinator. One decode trace stays resident and a finished
          request's slot is refilled at the next 16-step boundary, so a new
          request no longer waits for a whole batch to finish. Live streams pause
          while a refill's prompt is prefilled beside them. Bounds come from the
          package's continuous-slot policy; the admission receipts are
          pkgs/qwen36/evidence/2026-09-27/qwen38-continuous-slot-admission-receipt.json
          for the B4 profile and
          pkgs/qwen36/evidence/2026-10-21/qwen38-continuous-slot-admission-b8-receipt.json
          for the B8 profile.
        '';
      };

      continuousSlotProfile = mkOption {
        type = types.enum [
          "b4"
          "b8"
        ];
        default = "b4";
        description = ''
          Which continuous-slot scheduler profile the service selects when
          continuousSlotScheduler is enabled. "b4" is the four-slot profile and
          "b8" the eight-slot profile; both are admitted by their predeclared
          receipts, and every other value is refused before the service starts.
        '';
      };

      prefixCache = mkOption {
        type = types.bool;
        default = false;
        description = ''
          Keep hybrid prefix state for prompts that extend an earlier prompt, as an agent's
          next turn extends its conversation. A long prompt saves the carried GDN state at
          its last 1,024-token boundary and keeps the KV blocks before it; a later prompt
          with the same token prefix restores that state and prefills only the rest. Bounds
          come from the package's prefix-cache policy. Requires continuousSlotScheduler.
        '';
      };

      groupedPrefill = mkOption {
        type = types.bool;
        default = true;
        description = ''
          Prefill the users of one batch together, up to 256 prompt tokens each.
          Each forward serves a group of users rather than one. GDN layers run
          batched, and full-attention layers loop over the users. First-token
          logits and greedy tokens match per-user prefill exactly. Longer prompts,
          single users, and batches that reuse a parked decode trace keep
          per-user prefill.
        '';
      };

      tracedPrefill = mkOption {
        type = types.bool;
        default = true;
        description = ''
          Replay captured prefill traces: one trace per 128-, 256-, 512-, and
          1,024-token bucket, with the prompt length, position, and decode slot
          passed as device data. Longer prompts continue from the 1,024 bucket. Each replay also installs the user's
          decode state and logits, so a batch reads back once. Output is identical
          to eager prefill. The traces are captured at startup,
          after every decode width and eager prefill shape has been compiled.
          Startup therefore takes a few minutes longer. If a program first compiles
          after the capture, the traces are released and prefill stays eager until
          restart.
        '';
      };

      gdnFusedPrefill = mkOption {
        type = types.bool;
        default = false;
        description = ''
          Prefill GDN layers with the fused chunk op instead of the composite
          sequence path. Measured on P150x2 it halves short-prompt time to first
          token (2.2 to 1.1 seconds) at teacher-forced NLL +0.007 and top-1
          agreement 0.958 against the composite prefill. Opt-in only.
        '';
      };

      gdnProjectionBf4 = mkOption {
        type = types.bool;
        default = false;
        description = ''
          Store the GDN in and out projections as BF4. Measured on P150x2 with the
          fused decode step it cuts the decode step 5.9 percent at passage NLL
          +0.008. It changes whether the model opens a thinking block on some
          prompts. Opt-in only; the first start writes the BF4 weight cache.
        '';
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
        description = "Maximum combined prompt and generation tokens for one request. The default 131,072-token row uses 2,048 blocks of the package's generation KV range; two full-length requests cannot run alongside the reserved read lanes.";
      };

      maximumGenerationTokens = mkOption {
        type = types.ints.positive;
        default = defaultQwenMaximumGenerationTokens;
        description = "Maximum generated token count for one request, and the default when a request names none. Under the group coordinator a batch runs until its longest request finishes, so long generations hold the devices from other requests; the continuous-slot scheduler refills a slot as soon as its request ends.";
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

    ttModel = {
      enable = mkEnableOption "the tt-model owner helper and its single sudo grant";

      operatorUser = mkOption {
        type = types.str;
        default = "brittonr";
        description = "The one user allowed to run the owner helper without a password.";
      };

      ownerPackage = mkOption {
        type = types.package;
        default = pkgs.tt-model-owner;
        defaultText = lib.literalExpression "pkgs.tt-model-owner";
        description = "The owner helper package. Exactly one sudo rule names its binary.";
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
        {
          assertion = !(qwen38.mtpSpeculation && qwen38.sampledFusion);
          message = "MTP speculation cannot combine with sampled fusion";
        }
        {
          assertion = !(qwen38.mtpSpeculation && qwen38.gdnDecodeStep);
          message = "MTP speculation has no evidence on the fused GDN decode step; disable gdnDecodeStep";
        }
        {
          assertion = !qwen38.prefixCache || qwen38.continuousSlotScheduler;
          message = "The Qwen prefix cache needs the continuous-slot scheduler";
        }
        {
          assertion = !(qwen38.mtpSpeculation && qwen38.continuousSlotScheduler);
          message = "The continuous-slot scheduler needs the batch worker; MTP speculation serves the eager route";
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
          QWEN35_GDN_DECODE_FUSION = if qwen38.gdnDecodeStep then "step" else "off";
          QWEN35_GROUPED_PREFILL = if qwen38.groupedPrefill then "1" else "0";
          QWEN36_TRACED_PREFILL = if qwen38.tracedPrefill then "1" else "0";
          QWEN36_GDN_FUSED_PREFILL = if qwen38.gdnFusedPrefill then "1" else "0";
          QWEN36_GDN_BF4 = if qwen38.gdnProjectionBf4 then "1" else "0";
          QWEN36_SAMPLED_FUSION = if qwen38.sampledFusion then "1" else "0";
          QWEN36_MTP_SERVING = if qwen38.mtpSpeculation then "1" else "0";
          TT_CACHE_PATH = qwen38.modelPath;
          XDG_CACHE_HOME = "/var/lib/${qwenStateDirectory}/cache";
        };

        unitConfig.ConditionPathExists = [
          qwen38.modelPath
          "/dev/tenstorrent/0"
          "/dev/tenstorrent/1"
        ];

        serviceConfig = {
          ExecStart = lib.escapeShellArgs (
            [
              "${qwen38.package}/bin/qwen38-p150x2-serve"
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
              "--require-systemd-notify"
            ]
            ++ lib.optionals qwen38.continuousSlotScheduler [
              "--continuous-slot-policy"
              "${qwen38.package}/share/qwen36/${continuousSlotPolicyFile}"
            ]
            ++ lib.optionals qwen38.prefixCache [
              "--prefix-cache-policy"
              "${qwen38.package}/share/qwen36/prefix-cache-policy.json"
            ]
            ++ lib.optionals qwen38.mtpSpeculation [
              "--decode-mode"
              "eager"
              "--mtp-drafter-profile"
              "${qwen38.package}/share/qwen36/mtp-drafter-profile.json"
              "--mtp-trace-profile"
              "${qwen38.package}/share/qwen36/mtp-trace-profile.json"
            ]
          );
          Group = cfg.deviceGroup;
          NotifyAccess = "main";
          Restart = "on-failure";
          RestartSec = "${toString qwenRestartSeconds}s";
          StateDirectory = qwenStateDirectory;
          StateDirectoryMode = qwenStateDirectoryMode;
          TimeoutStartSec = 0;
          TimeoutStopSec = qwenStopTimeoutSeconds;
          Type = "notify";
          UnsetEnvironment = [
            "TT_MESH_GRAPH_DESC_PATH"
            "TT_METAL_SIMULATOR"
          ];
          User = "root";
          WorkingDirectory = "/var/lib/${qwenStateDirectory}";
        };
      };
    })

    (mkIf cfg.ttModel.enable (
      let
        helper = "${cfg.ttModel.ownerPackage}/bin/tt-model-owner";
        helperPackage = cfg.ttModel.ownerPackage;
        helperRules = lib.filter (
          rule: lib.any (command: command.command == helper) rule.commands
        ) config.security.sudo.extraRules;
        # Any grant for the operator that resolves inside the helper package but is
        # not the helper binary itself is wider than the single command this module
        # is allowed to grant.
        widerCommands = lib.concatMap (
          rule:
          if !(lib.elem cfg.ttModel.operatorUser rule.users) then
            [ ]
          else
            lib.filter (
              command:
              command.command == "${helperPackage}/bin"
              || (lib.hasPrefix "${helperPackage}/" command.command && command.command != helper)
            ) rule.commands
        ) config.security.sudo.extraRules;
      in
      {
        assertions = [
          {
            assertion = cfg.ttModel.operatorUser != "" && cfg.ttModel.operatorUser != "ALL";
            message = "hardware.tenstorrent.ttModel.operatorUser must name one user, not ALL or an empty string";
          }
          {
            assertion =
              lib.length helperRules == 1
              && (lib.head helperRules).users == [ cfg.ttModel.operatorUser ]
              &&
                (lib.head helperRules).commands == [
                  {
                    command = helper;
                    options = [ "NOPASSWD" ];
                  }
                ];
            message = "hardware.tenstorrent.ttModel must render exactly one NOPASSWD rule for ${helper} and no other command";
          }
          {
            assertion = widerCommands == [ ];
            message = "hardware.tenstorrent.ttModel grants ${
              lib.concatStringsSep ", " (map (command: command.command) widerCommands)
            } which is wider than the single helper binary";
          }
        ];

        security.sudo.extraRules = [
          {
            users = [ cfg.ttModel.operatorUser ];
            commands = [
              {
                command = helper;
                options = [ "NOPASSWD" ];
              }
            ];
          }
        ];

        # The root-owned record, its roots, and the pointer that tells the helper
        # which installed copy may act. This does not enable or unmask any unit.
        systemd.tmpfiles.rules = [
          "d /var/lib/tenstorrent-models 0755 root root -"
          "d /nix/var/nix/gcroots/tenstorrent-models 0755 root root -"
          "L+ /var/lib/tenstorrent-models/owner-helper - - - - ${helperPackage}"
        ];
      }
    ))

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
