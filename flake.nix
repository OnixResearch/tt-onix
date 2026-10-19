{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs";
    systems.url = "github:nix-systems/default-linux";
    flake-parts.url = "github:hercules-ci/flake-parts";
    treefmt-nix.url = "github:numtide/treefmt-nix";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    cairn.url = "github:OnixResearch/cairn/15f00875562025e7ea7e0d1f4af24d1a2e2ac06f";
  };

  outputs =
    { self, ... }@inputs:
    inputs.flake-parts.lib.mkFlake { inherit inputs; } {
      systems = import inputs.systems;

      imports = [
        inputs.treefmt-nix.flakeModule
        ./flake-modules/onix-packages.nix
      ];

      flake = {
        overlays.default = import ./overlay.nix;
        nixosModules = {
          default = ./nixos/module.nix;
          "dual-p150a-training" = ./nixos/examples/dual-p150a-training.nix;
          "qwen38-p150x2-read-lanes" = ./nixos/examples/qwen38-p150x2-read-lanes.nix;
        };
      };

      perSystem =
        { system, pkgs, ... }:
        let
          metaliumRsToolchain = pkgs.rust-bin.stable."1.97.1".default.override {
            targets = [ "riscv32im-unknown-none-elf" ];
            extensions = [ "llvm-tools" ];
          };
          metaliumRs = pkgs.callPackage ./pkgs/metalium-rs {
            rustToolchain = metaliumRsToolchain;
          };
          metaliumRsCheckpoint = pkgs.fetchurl {
            name = "rwkv7-goose-world2.8-0.1b-d81965cb4e1a9f96696b4f70b84212b8f2e43216.safetensors";
            url = "https://huggingface.co/RWKV/RWKV7-Goose-World2.8-0.1B-HF/resolve/d81965cb4e1a9f96696b4f70b84212b8f2e43216/model.safetensors";
            hash = "sha256-uWqL3CHhX3HgyVZT3MO+ieVkthmtUHPJ7b+9B/eElFM=";
          };
          metaliumRsCheckpointGate = pkgs.writeShellApplication {
            name = "rwkv7-layer-checkpoint";
            text = ''
              exec ${metaliumRs}/bin/rwkv7-layer-gate --checkpoint ${metaliumRsCheckpoint} "$@"
            '';
          };
        in
        {
          _module.args.pkgs = import inputs.nixpkgs {
            inherit system;
            overlays = [
              (import inputs.rust-overlay)
              self.overlays.default
            ];
          };

          treefmt = {
            programs.nixfmt.enable = true;
            settings.formatter.nixfmt.excludes = [
              "quality/fixtures/language/nix/malformed.nix"
            ];
          };

          packages = {
            inherit (pkgs)
              enchantum
              llama-cpp-metalium
              luwen
              nanobench
              tt-burnin
              tt-logger
              tt-metal
              tt-smi
              tt-system-tools
              tt-topology
              tt-umd
              ttsim
              ttsim-blackhole-smoke
              ttsim-runtime-validator
              ;
            inherit (pkgs.python3Packages)
              pyluwen
              tt-flash
              tt-tools-common
              ;
            inherit (pkgs.linuxPackagesFor pkgs.linux) tt-kmd;
            metalium-rs = metaliumRs;
            metalium-rs-rwkv7-checkpoint = metaliumRsCheckpointGate;
          }
          // pkgs.lib.optionalAttrs (system == "x86_64-linux") {
            inherit (pkgs) qwen38-trace-reuse-host1 qwen36-trace-reuse-host1;
          };

          apps.ttsim-blackhole-smoke = {
            type = "app";
            program = "${pkgs.ttsim-blackhole-smoke}/bin/ttsim-blackhole-smoke";
            meta.description = "Run the bounded Blackhole 1x1 TT-NN smoke on TTsim";
          };

          apps.metalium-rs-mnist = {
            type = "app";
            program = "${metaliumRs}/bin/tt-mnist";
            meta.description = "Train MNIST with the native Rust Blackhole stack";
          };

          apps.metalium-rs-wkv7 = {
            type = "app";
            program = "${metaliumRs}/bin/wkv7-gate";
            meta.description = "Run the native RWKV-7 WKV7 state-carry gate";
          };

          apps.metalium-rs-rwkv7-layer = {
            type = "app";
            program = "${metaliumRs}/bin/rwkv7-layer-gate";
            meta.description = "Compare one RWKV-7 layer on Flex and TtBackend";
          };

          apps.metalium-rs-rwkv7-checkpoint = {
            type = "app";
            program = "${metaliumRsCheckpointGate}/bin/rwkv7-layer-checkpoint";
            meta.description = "Compare the real RWKV-7 layer zero on Flex and TtBackend";
          };

          checks = {
            inherit (pkgs)
              enchantum
              llama-cpp-metalium
              luwen
              nanobench
              tt-burnin
              tt-logger
              tt-metal
              tt-smi
              tt-system-tools
              tt-topology
              tt-umd
              ttsim
              ttsim-runtime-validator
              ;
            ttsim-runtime-bundle = pkgs.ttsim-check;
            ttsim-single-chip-smoke = pkgs.ttsim-smoke-check;
            ttsim-gdn-reference-smoke = pkgs.ttsim-gdn-smoke-check;
            ttsim-gdn-input-validation = pkgs.ttsim-gdn-validation-check;
            inherit (pkgs.python3Packages)
              pyluwen
              tt-flash
              tt-tools-common
              ;
            inherit (pkgs.linuxPackagesFor pkgs.linux) tt-kmd;
            metalium-rs = metaliumRs;
          }
          // pkgs.lib.optionalAttrs (system == "x86_64-linux") {
            inherit (pkgs) qwen38-trace-reuse-host1 qwen36-trace-reuse-host1;
          };
        };
    };
}
