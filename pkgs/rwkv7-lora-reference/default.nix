{
  lib,
  nickel,
  python3,
  runCommand,
  model,
  modelBlake3,
}:
let
  pythonEnvironment = python3.withPackages (packages: [
    packages.blake3
    packages.safetensors
    packages.torch
  ]);
  source = ./.;
in
runCommand "rwkv7-lora-reference-check"
  {
    nativeBuildInputs = [
      nickel
      pythonEnvironment
    ];
    RWKV7_LORA_MODEL = model;
    RWKV7_LORA_MODEL_BLAKE3 = modelBlake3;
  }
  ''
    set -euo pipefail
    cp -r ${source} "$PWD/source"
    chmod -R u+w "$PWD/source"
    cd "$PWD/source"

    # Configuration admission: valid runner exports deterministically.
    nickel export --format json runner.ncl >runner-first.json
    nickel export --format json runner.ncl >runner-second.json
    cmp runner-first.json runner-second.json
    grep -F '"model_id": "RWKV/RWKV7-Goose-World2.8-0.1B-HF"' runner-first.json

    # Negative configuration with an empty LoRA target set must fail.
    if nickel export --format json fixtures/invalid-config-empty-targets.ncl \
        >invalid-config.json 2>invalid-config.log; then
      echo "Nickel accepted a configuration with an empty LoRA target set" >&2
      exit 1
    fi

    # Device-free bounded reference validation.
    python reference.py >receipt.json
    grep -F '"outcome": "bounded_cpu_reference_pass"' receipt.json
    grep -F '"base_frozen": true' receipt.json
    grep -F '"adapter_changed": true' receipt.json
    grep -F '"resume_rejected": true' receipt.json
    grep -F '"checkpoint_dtype": "bf16"' receipt.json
    grep -F '"fp32_from_bf16_finite": true' receipt.json
    grep -F 'No Tenstorrent execution is established.' receipt.json

    mkdir -p "$out"
    cp receipt.json "$out/receipt.json"
    cp runner-first.json "$out/runner.json"
    cp README.md "$out/README.md"
  ''
