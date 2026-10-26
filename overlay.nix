final: prev: {
  enchantum = final.callPackage ./pkgs/enchantum { };
  llama-cpp-metalium = final.callPackage ./pkgs/llama-cpp-metalium { };
  luwen = final.callPackage ./pkgs/luwen { };
  tt-metal-tracy-tools = final.callPackage ./pkgs/tt-metal-tracy-tools { };
  nanbeige43 = final.callPackage ./pkgs/nanbeige43 { };
  qwen38 = final.callPackage ./pkgs/qwen36 { };
  qwen38-trace-reuse-host1 = final.callPackage ./pkgs/qwen36-trace-reuse-host1 { };
  # Compatibility aliases. The checkpoint is Qwen3.8-27B, not Qwen3.6.
  qwen36 = final.qwen38;
  qwen36-trace-reuse-host1 = final.qwen38-trace-reuse-host1;
  xtensor-blas = prev.xtensor-blas.overrideAttrs (prevAttrs: {
    postFixup = (prevAttrs.postFixup or "") + ''
      sed -i "s|\''${PACKAGE_PREFIX_DIR}//nix/store|/nix/store|g" $out/lib/cmake/xtensor-blas/xtensor-blasConfig.cmake
    '';
  });
  nanobench = prev.nanobench.overrideAttrs (prevAttrs: {
    patches = (prevAttrs.patches or [ ]) ++ [ ./pkgs/nanobench/fix-cmake-find_package.patch ];
  });
  tt-burnin = final.callPackage ./pkgs/tt-burnin { };
  tt-logger = final.callPackage ./pkgs/tt-logger { };
  tt-metal = final.callPackage ./pkgs/tt-metal { };
  # GDN grid instance cap lifted; opt-in variant, not the shared runtime.
  tt-metal-gdn-grid = final.callPackage ./pkgs/tt-metal-gdn-grid { };
  tt-smi = final.callPackage ./pkgs/tt-smi { };
  tt-system-tools = final.callPackage ./pkgs/tt-system-tools { };
  tt-topology = final.callPackage ./pkgs/tt-topology { };
  tt-umd = final.callPackage ./pkgs/tt-umd { };
  ttsim = final.callPackage ./pkgs/ttsim { };
  tt-lang-sim = final.callPackage ./pkgs/tt-lang-sim { };
  tenstorrent-official-skills = final.callPackage ./pkgs/tenstorrent-official-skills { };
  tenstorrent-eval-tools = final.callPackage ./pkgs/tenstorrent-eval-tools { };
  ttsim-check = final.callPackage ./pkgs/ttsim/check.nix { };
  ttsim-runtime-validator = final.callPackage ./pkgs/ttsim-runtime-validator { };
  ttsim-blackhole-smoke = final.callPackage ./pkgs/ttsim/smoke.nix { };
  ttsim-smoke-check = final.callPackage ./pkgs/ttsim/smoke-check.nix { };
  ttsim-gdn-smoke-check = final.callPackage ./pkgs/ttsim/gdn-smoke-check.nix { };
  ttsim-gdn-validation-check = final.callPackage ./pkgs/ttsim/gdn-smoke-check.nix {
    validationOnly = true;
  };

  pythonPackagesExtensions = prev.pythonPackagesExtensions ++ [
    (pyfinal: pyprev: {
      pyluwen = pyfinal.callPackage ./pkgs/pyluwen { };
      tt-flash = pyfinal.callPackage ./pkgs/tt-flash { };
      tt-tools-common = pyfinal.callPackage ./pkgs/tt-tools-common { };
    })
  ];

  linuxKernel = prev.linuxKernel // {
    packagesFor =
      kernel:
      (prev.linuxKernel.packagesFor kernel).extend (
        lpfinal: lpprev: {
          tt-kmd = lpfinal.callPackage ./pkgs/tt-kmd { };
        }
      );
  };
}
