# Cloudflare Clef released artifacts; URLs and content hashes are pinned at the two revisions.
{
  lib,
  fetchurl,
  runCommand,
}:
let
  mkModel =
    {
      variant,
      revision,
      files,
    }:
    let
      base = "https://huggingface.co/Cloudflare/${variant}/resolve/${revision}";
      fetched = lib.mapAttrs (
        filename: hash:
        fetchurl {
          url = "${base}/${filename}";
          name = "${variant}-${filename}";
          inherit hash;
        }
      ) files;
    in
    runCommand "cloudflare-${variant}-${builtins.substring 0 12 revision}" { } (
      ''
        mkdir -p "$out"
      ''
      + lib.concatStringsSep "\n" (
        lib.mapAttrsToList (filename: source: "ln -s ${source} \"$out/${filename}\"") fetched
      )
      + "\n"
      + ''
        printf '%s\n' '${revision}' > "$out/revision"
      ''
    );
in
{
  clef = mkModel {
    variant = "clef";
    revision = "2f3de3dd85f379784083b0814d997ab627200f0c";
    files = {
      "LICENSE" = "sha256-u+3D/aMwWCC5dyZfAbhhnYdXCmc53jpVgsNGSEDx5Xo=";
      "chat_template.jinja" = "sha256-w8+eNKv0+eNsLXIWWqnBMtPipyW2wlhqqjqK+deoEEE=";
      "config.json" = "sha256-xC6IiSvT/YToJ2sq2Q31jBw7eXZ26hYQNQBqcq1GjFg=";
      "generation_config.json" = "sha256-3DaFSyslF2dkz+7aMqPYD7T8/77ysA1VxmFAeb8h658=";
      "joint_head.safetensors" = "sha256-oBCsBPB45pmYjkBJy+peYsliOT9Z/sNmZAtk6NaaSVM=";
      "joint_head_config.json" = "sha256-iQvlhddbmBIB65ajX5iplnr+N7yNIgz96i5y1WUHU08=";
      "joint_schema_model.py" = "sha256-DjBM98ZQDou1m+9+Kv0sY3P4JZbfs7V9Gqk8F14tw6M=";
      "model.safetensors.index.json" = "sha256-pgDGYm6x62U6PweKs90ceo+rLnEYXH7zqpI6OaqmN74=";
      "processor_config.json" = "sha256-2J70nOnNN/v1EBWOE8HvBj2ShkEcHskEmTLb4EhxQ7E=";
      "tokenizer.json" = "sha256-BrlQk1LSr1A4GrIkfgg7gNMtXAq6kcJyyp/3Kbag5SM=";
      "tokenizer_config.json" = "sha256-kaCPgl03DQhdaS4EzxF83X+q178Y6Zbx5gMbbasD23I=";
      "model-00001-of-00012.safetensors" = "sha256-VNg8HTZjHeIxh2IXqODCSD7M7odGNppIK3lEK9/F2Vg=";
      "model-00002-of-00012.safetensors" = "sha256-RkCGrwi+ji7BSWCk3P8IPrw5l0reANedNUlzhflgqzo=";
      "model-00003-of-00012.safetensors" = "sha256-CSIS06Avr6zWQkcj7aWdYOUoLSBo1o8ON8uJH2O7tlg=";
      "model-00004-of-00012.safetensors" = "sha256-0G/xl2aMeCFF+vp0u6YbvClvsn45r7Ff1SKRjONRTcU=";
      "model-00005-of-00012.safetensors" = "sha256-zGk7iClhSnLgwr4wP7KQzwXbtNfe2HKoF7l7siKnhCc=";
      "model-00006-of-00012.safetensors" = "sha256-58zhXaJEPLi4Sq7Wappx8Mh9ydBD+DtfWPSon2S6YK0=";
      "model-00007-of-00012.safetensors" = "sha256-dfx+drV9XRel2F//PoedB90z7ciFqO4ErUN6iZvNUwc=";
      "model-00008-of-00012.safetensors" = "sha256-GJsVy2sa9I1fEYlRRG4VY5v+r3YIHV8grtHxtCU6/h0=";
      "model-00009-of-00012.safetensors" = "sha256-gQHiZkuxRoT8cFHy4fhJA9vfVImxfK4yEqwIoK90SmA=";
      "model-00010-of-00012.safetensors" = "sha256-qOaQFqGo2rHJzo3RUcDTIkQSq4ZM8G06XiuKV3XP37g=";
      "model-00011-of-00012.safetensors" = "sha256-syjSHDarOEaW5AowrIapWq9t2C2olDi6AJy5fYfB1rk=";
      "model-00012-of-00012.safetensors" = "sha256-dQXu2RCoTqGOZpU1ckdviopqbvVBkqZkPW6c0ho7uXg=";
    };
  };
  clef-flash = mkModel {
    variant = "clef-flash";
    revision = "17f0b0ad64efb65d273590632833508766b2aae6";
    files = {
      "LICENSE" = "sha256-u+3D/aMwWCC5dyZfAbhhnYdXCmc53jpVgsNGSEDx5Xo=";
      "chat_template.jinja" = "sha256-pK7or88uBxGULPhIiZvmYBb40UqIn/nt4HvKCZwo9xU=";
      "config.json" = "sha256-Zvh/b7Jha0ZgTa8qnGfdyHk4KW0HFW76NNWbW+SeMjg=";
      "generation_config.json" = "sha256-RXB/hGe/TlThEsYJWhstHy1jZRz4aBY0TP/yHwLdoNQ=";
      "joint_head.safetensors" = "sha256-Gc3OyMgdySEr4yD/9HRiqzQvvBJ4vkNo+z2nEkHPW6A=";
      "joint_head_config.json" = "sha256-d+/pWaOLWxeyQVQ+Ep5pXzx3Rl7OVaJZhSeb2BdieaA=";
      "joint_schema_model.py" = "sha256-DjBM98ZQDou1m+9+Kv0sY3P4JZbfs7V9Gqk8F14tw6M=";
      "model.safetensors.index.json" = "sha256-lBMF/593VR4UWmzql270Vs1csgjL7OmfFoN2x1L8+Ww=";
      "processor_config.json" = "sha256-2J70nOnNN/v1EBWOE8HvBj2ShkEcHskEmTLb4EhxQ7E=";
      "tokenizer.json" = "sha256-BrlQk1LSr1A4GrIkfgg7gNMtXAq6kcJyyp/3Kbag5SM=";
      "tokenizer_config.json" = "sha256-kaCPgl03DQhdaS4EzxF83X+q178Y6Zbx5gMbbasD23I=";
      "model-00001-of-00004.safetensors" = "sha256-i0Wo6WgUHNzFj7ccmt/CWOLHe18GK8Y2wf1bxdkWtWU=";
      "model-00002-of-00004.safetensors" = "sha256-dZCFbHE+7YRKLc9I5sQ8TeFlt4i8P4DjKDERg828fbg=";
      "model-00003-of-00004.safetensors" = "sha256-5urCRnlSwzNh7X3LPHlZ0QhrvlcgHNN0nD12n9wX/mM=";
      "model-00004-of-00004.safetensors" = "sha256-n87MZVazkXEjg3OkZfQJeUt/gh+0zR5kWeOpwP4xevc=";
    };
  };
}
