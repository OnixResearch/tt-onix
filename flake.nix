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
          }
          // pkgs.lib.optionalAttrs (system == "x86_64-linux") {
            inherit (pkgs) qwen38-trace-reuse-host1 qwen36-trace-reuse-host1;
          };

          apps.ttsim-blackhole-smoke = {
            type = "app";
            program = "${pkgs.ttsim-blackhole-smoke}/bin/ttsim-blackhole-smoke";
            meta.description = "Run the bounded Blackhole 1x1 TT-NN smoke on TTsim";
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
          }
          // pkgs.lib.optionalAttrs (system == "x86_64-linux") {
            inherit (pkgs) qwen38-trace-reuse-host1 qwen36-trace-reuse-host1;
          };
        };
    };
}
