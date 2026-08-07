{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs";
    systems.url = "github:nix-systems/default-linux";
    flake-parts.url = "github:hercules-ci/flake-parts";
    treefmt-nix.url = "github:numtide/treefmt-nix";
    cairn.url = "github:OnixResearch/cairn/837d730799faffe6a088cfd6dc49e08c7e0f1822";
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
        nixosModules.default = ./nixos/module.nix;
      };

      perSystem =
        { system, pkgs, ... }:
        {
          _module.args.pkgs = import inputs.nixpkgs {
            inherit system;
            overlays = [
              self.overlays.default
            ];
          };

          treefmt.programs = {
            nixfmt.enable = true;
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
            inherit (pkgs.python3Packages)
              pyluwen
              tt-flash
              tt-tools-common
              ;
            inherit (pkgs.linuxPackagesFor pkgs.linux) tt-kmd;
          };
        };
    };
}
