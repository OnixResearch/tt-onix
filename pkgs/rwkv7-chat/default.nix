{
  lib,
  writeShellApplication,
  fetchurl,
  llama-cpp-metalium,
}:
let
  # RWKV7-Goose-World2.8-0.1B in GGUF form. This is the same base model the
  # repository pins as safetensors for the P150x2 runtime, mirrored to GGUF by
  # mradermacher so llama.cpp can load it. Q8_0 quantization.
  model = fetchurl {
    name = "rwkv7-goose-world2.8-0.1b-q8_0.gguf";
    url = "https://huggingface.co/mradermacher/RWKV7-Goose-World2.8-0.1B-HF-GGUF/resolve/main/RWKV7-Goose-World2.8-0.1B-HF.Q8_0.gguf";
    hash = "sha256-kQuW42a2sy3950Ta2iaMosV5tfERYKwo495IfrxYwEQ=";
  };
  llamaCli = "${llama-cpp-metalium}/bin/llama-cli";
in
writeShellApplication {
  name = "rwkv7-chat";
  text = ''
    # Single-card Metalium run. The system exports TT_MESH_GRAPH_DESC_PATH for a
    # two-card p150_x2 mesh, which cannot map onto one card, so drop it.
    # GGML_METALIUM_DISABLE_GRAPH_COMPILER avoids the graph-compiler lerp fusion
    # path, which JITs a tt-metal kernel that needs a header absent from the
    # packaged tt-metal 0.74 runtime include tree.
    if [ "$#" -eq 0 ]; then
      printf 'usage: rwkv7-chat PROMPT [llama-cli args...]\n' >&2
      exit 2
    fi
    prompt="$1"
    shift
    exec env -u TT_MESH_GRAPH_DESC_PATH \
      GGML_METALIUM_DISABLE_GRAPH_COMPILER=1 \
      GGML_METALIUM_DEVICE_ID=0 \
      TT_VISIBLE_DEVICES=0 \
      ${lib.escapeShellArg llamaCli} \
      -m ${model} \
      -ngl 99 \
      --no-warmup \
      -p "$prompt" \
      "$@"
  '';
}
