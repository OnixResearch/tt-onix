# AGENTS.md

## Local pi skills

The `.pi/skills/` directory holds pi skills copied from the reference checkouts.
The `ttxla-*` skills come from `tt-xla/.claude/skills` at the pinned commit.
The `ttllk-*` skills come from `tt-metal-latest/tt_metal/tt-llk/.claude/skills`.
Skill `name:` fields match their directory names, so `/skill:ttxla-graph-break-analysis` and `/skill:ttllk-race-audit-all` work.
Refresh a skill by copying the upstream directory again after a reference update.

## Tenstorrent documentation references

Read these local references before you change Tenstorrent packages, modules, device setup, kernels, or operators.

- Read the downloaded public site source under `.pi/references/tenstorrent/docs-site`.
  Start with `.pi/references/tenstorrent/docs-site/core` for the main site.
  The other build roots are `ttnn`, `tt-metalium`, `syseng`, and `cloud-native-support`.
  Source: <https://github.com/tenstorrent/tenstorrent.github.io>.
  The downloaded checkout uses commit `c37781bf680fc269cfd31097e2e3e47cc54aa414`.
- Use `.pi/references/tenstorrent/docs-tenstorrent-com-index.html` for the downloaded public landing page.
  Use `.pi/references/tenstorrent/docs-tenstorrent-com-index.rst.txt` for its readable source text.
  Source: <https://docs.tenstorrent.com/>.
- Read the package-matched `tt-metal` docs at `.pi/references/tenstorrent/tt-metal-docs/docs`.
  Source: <https://github.com/tenstorrent/tt-metal/tree/v0.74.0-rc6/docs>.
  The downloaded checkout uses commit `c49bb7625e6a495337483353a5228339fece2c11`.
- Read the built latest TT-NN documentation at `.pi/references/tenstorrent/ttnn-docs-latest/docs.tenstorrent.com/tt-metal/latest/ttnn`.
  Start at `index.html`.
  Source: <https://docs.tenstorrent.com/tt-metal/latest/ttnn/index.html>.
  The downloaded mirror uses the site state of 2026-08-05.
- Read the technical reports at `.pi/references/tenstorrent/tt-metal-docs/tech_reports`.
  Source: <https://github.com/tenstorrent/tt-metal/tree/v0.74.0-rc6/tech_reports>.

Use the public docs for current product and installation information.
Use the package-matched docs for behavior that depends on the local `tt-metal` package.
If documentation and packaged source differ, inspect the packaged source and record the mismatch.
Do not update a package pin only because the current public docs changed.

### Tenstorrent implementation references

Read these package-matched or commit-pinned sources before you change hardware behavior or host-device interfaces.

- Read the package-matched implementation paths under `.pi/references/tenstorrent/tt-metal-docs`.
  The checkout includes `tt_metal`, `ttnn/cpp`, `tests/tt_metal`, and `tests/ttnn`.
  Source: <https://github.com/tenstorrent/tt-metal/tree/v0.74.0-rc6>.
- Read the ISA contract at `.pi/references/tenstorrent/tt-isa-documentation`.
  Source: <https://github.com/tenstorrent/tt-isa-documentation>.
  The downloaded checkout uses commit `b7738d9ac14a34a4033d60dde9463466b23082e1`.
- Read the LLK, NoC, overlay, and performance docs at `.pi/references/tenstorrent/tt-low-level-documentation`.
  Source: <https://github.com/tenstorrent/tt-low-level-documentation>.
  The downloaded checkout uses commit `cf344d2dda4cb9d06451d0cc62e9a12c7f1b9240`.
- Read the host-driver sources at `.pi/references/tenstorrent/tt-umd` and `.pi/references/tenstorrent/tt-kmd`.
  The `tt-umd` checkout uses tag `v0.9.9` and commit `8f2f3d293852ee4f129eea5a40f6c15f7d259add`.
  The `tt-kmd` checkout uses tag `ttkmd-2.10.0` and commit `49ae9d88e177635746db4c085bd21a9481adb591`.
- Read the device-free simulator source and ABI docs at `.pi/references/tenstorrent/ttsim`.
  Source: <https://github.com/tenstorrent/ttsim/tree/v1.10.1>.
  The downloaded checkout uses commit `3d82cd1b8e3c3d8fa69abb8ac3835c4713e4e52a`.
- Read the SFPI compiler source at `.pi/references/tenstorrent/sfpi`.
  Source: <https://github.com/tenstorrent/sfpi/tree/7.70.0>.
  The downloaded checkout uses commit `6659aeb9e350dd8eef69edbdfa4f963759b04c44`.
  The tt-metal 0.74 source names SFPI 7.64.0. The Nix package intentionally carries 7.70.0 and rewrites the installed version metadata.
- Read the training-platform frontend source at `.pi/references/tenstorrent/tt-xla`.
  Source: <https://github.com/tenstorrent/tt-xla>.
  The downloaded checkout uses commit `10b111f3885a598958b0c4e393a07d672584736a`.
  This commit matches the `tt_xla_source` pin in `pkgs/training-platform/manifest.ncl`.
- Read the latest upstream `tt-metal` source at `.pi/references/tenstorrent/tt-metal-latest`.
  Source: <https://github.com/tenstorrent/tt-metal>.
  The downloaded checkout uses commit `0fa87dd38f087a10383dd320670419428db45979` from the `main` branch on 2026-08-05.

Use the ISA documentation as the hardware behavior contract.
Use the low-level documentation for kernel API and NoC guidance.
Use the package-matched driver sources for host ABI and device-node behavior.
Use `ttsim` for device-free behavior checks when its documented feature set covers the path.
Use `tt-xla` for PJRT, PyTorch/XLA, and JAX frontend behavior before you change the training platform.
Use `tt-metal-latest` and the latest TT-NN mirror for current upstream behavior only.
The package-matched `tt-metal-docs` checkout stays the authority for packaged behavior.
Do not update a package pin only because the latest upstream source changed.

### Simulator-before-hardware workflow

Use a short `ttsim` smoke test before a new hardware attempt when the simulator covers the required architecture and operations.
Do not wait for full simulated inference when a smaller test can remove the same software risk.

- Run the same host and device binaries on `ttsim` and silicon. Do not add simulator-specific runtime branches.
- Start Blackhole validation with the packaged `libttsim_bh.so` and a `1x1` mesh.
  Require one device open, firmware compilation, representative weight loading, one RWKV layer, and recurrent-state carry.
- Use a hardware-neutral smoke mode when the production runtime is fixed to a larger mesh.
  Keep the production P150x2 command fail-closed on an exact `1x2` mesh with physical device IDs 0 and 1.
- Advance to the packaged `libttsim_bh_x2.so` only after the single-chip smoke test passes.
  Use a package-matched cluster descriptor, and test mesh opening, replication, and the bounded representative path.
  Treat this experimental P300 model only as a preliminary check for a two-P150 deployment.
- Prefer slow dispatch because `ttsim` fast dispatch has incomplete determinism coverage.
- Set `TT_METAL_DISABLE_SFPLOADMACRO=1` when the tested SFPU path can use `SFPLOADMACRO`.
- Use a named 60-minute timeout for a full-model simulator attempt. Stop at the timeout and preserve the exact blocker.
- Require the smoke test to open the expected mesh, compile firmware, execute representative TT-NN operations, and carry recurrent state.
- Mark every simulator receipt as simulated. Never use simulator evidence to claim physical placement, hardware profiling, performance, or successful silicon execution.
- If the exact simulator path needs an unmatched TT-Metal branch or unsupported feature, record the mismatch and move to one bounded hardware attempt.
- After the simulator smoke test passes, use hardware for physical-device attribution, profiler evidence, performance, and the final acceptance result.

### Tenstorrent system-management references

- Read the package-matched Luwen source at `.pi/references/tenstorrent/luwen`.
  Source: <https://github.com/tenstorrent/luwen/tree/v0.9.0>.
  The downloaded checkout uses commit `8d9175cc71da8c13cf86f46264f7468884f800d8`.

Use Luwen for chip discovery, initialization, diagnostics, firmware communication, and Rust host-tooling behavior.

### Tenstorrent operations references

- Read the shared tool source at `.pi/references/tenstorrent/tt-tools-common`.
  Source: <https://github.com/tenstorrent/tt-tools-common/tree/v1.6.0>.
  The downloaded checkout uses commit `dba5f77b5d9f29163fe1b971d5c79c7d89a47a3b`.
- Read the package-matched diagnostics sources at `.pi/references/tenstorrent/tt-smi` and `.pi/references/tenstorrent/tt-topology`.
  The `tt-smi` checkout uses tag `v6.2.1` and commit `07208d7e783b8bb585440460db53f515b5828f16`.
  The `tt-topology` checkout uses tag `v1.2.20` and commit `7bd675f888f9f0d27674fab7cb4188d5b1255481`.
- Read the firmware update tool at `.pi/references/tenstorrent/tt-flash`.
  Source: <https://github.com/tenstorrent/tt-flash/tree/v3.10.0>.
  The downloaded checkout uses commit `3e9f69a14c2b60a54ee645863486b5d57bdbaf6a`.
- Read the device stress tool at `.pi/references/tenstorrent/tt-burnin`.
  Source: <https://github.com/tenstorrent/tt-burnin/tree/v0.4.4>.
  The downloaded checkout uses commit `e9124ab38517ac847607afb6720fdadd7a5faf30`.
- Read the incident collection tools at `.pi/references/tenstorrent/tt-system-tools`.
  Source: <https://github.com/tenstorrent/tt-system-tools/tree/v1.4.1>.
  The downloaded checkout uses commit `8cea822ea86ba7336c16f164a3a9f939e2eb4392`.

Use these package-matched sources for command behavior, supported options, diagnostics, and failure handling.
Do not cache firmware binaries in the reference tree.

### Tenstorrent ecosystem references

- Read the curated project directory at `.pi/references/tenstorrent/tt-awesome`.
  Start with `README.md`. The structured entries live under `entries/`.
  Source: <https://github.com/tenstorrent/tt-awesome>.
  The downloaded checkout uses commit `a975e492a8e1826952b1967c96f2ead9573bfbab`.

Use `tt-awesome` to discover community and official projects, kernels, and tools for Tenstorrent hardware.
Treat its entries as leads, not as verified behavior. Confirm claims against the linked project source.
Do not treat a listed project as package-matched authority.

## RWKV-7 references

Read these local references before you change RWKV-7 code, equations, tests, or package pins.

- Read the extracted architecture paper at `.pi/references/rwkv-7/rwkv-7-goose-arxiv-2503.14456v2.txt`.
  Use `.pi/references/rwkv-7/rwkv-7-goose-arxiv-2503.14456v2.pdf` for figures and the original page layout.
  Source: <https://arxiv.org/abs/2503.14456>.
- Read the upstream source at `.pi/references/rwkv-7/RWKV-LM`.
  Source: <https://github.com/BlinkDL/RWKV-LM/tree/main>.
  The downloaded checkout uses commit `952102498e9ed367ea0a59ee64106916d474d30f`.
- Use `.pi/references/rwkv-7/oxen-how-rwkv-7-goose-works.html` for explanatory context and author notes.
  Source: <https://www.oxen.ai/blog/how-rwkv-7-goose-works-notes-from-the-author>.
- Use <https://deepwiki.com/BlinkDL/RWKV-LM/1-overview> for a generated repository overview and source-file index.

Treat the paper and upstream source as primary references.
Treat the DeepWiki overview and Oxen article as secondary explanations.
Check DeepWiki claims against the primary references because its indexed revision can differ.
Keep the existing package pins unless the task requires a pin update.
Record a mismatch when the paper, source, and local implementation differ.

### RWKV ecosystem references

Use these resources to find current RWKV projects and implementation examples.

- Use the [recently updated RWKV repository search](https://github.com/search?o=desc&q=rwkv&s=updated&type=Repositories) for project discovery.
- Use the [RWKV Wiki](https://wiki.rwkv.com) for community documentation about architectures, models, terminology, and tools.
- Use [josStorer/RWKV-Runner](https://github.com/josStorer/RWKV-Runner) as a desktop runner and OpenAI-compatible service reference.
- Use [JL-er/RWKV-PEFT](https://github.com/JL-er/RWKV-PEFT) as a parameter-efficient fine-tuning reference.
- Use [cgisky1980/ai00_rwkv_server](https://github.com/cgisky1980/ai00_rwkv_server) as a Rust and Vulkan inference-server reference.

Treat these resources as ecosystem and implementation leads.
Check architecture claims against the RWKV paper and `RWKV-LM` source.
Do not update a package pin only because an ecosystem resource changed.

### llama.cpp Metalium upstream reference

Treat <https://github.com/marty1885/llama.cpp> as the upstream source for the local `llama-cpp-metalium` package.

- Read the all-branch clone at `.pi/references/llama.cpp-marty1885`.
- Read `.pi/references/llama.cpp-marty1885-index/UPSTREAM-REFERENCE.md` before you select a branch or copy code.
- Use `.pi/references/llama.cpp-marty1885-index/BRANCHES.tsv` for the recorded branch tips.
- Use `.pi/references/llama.cpp-marty1885-index/TREE-SEARCH-SUMMARY.tsv` and `tree-hits/` for the complete tree-path search.

The package uses branch `metalium-support` at commit `04602ca95c1a6adb6c1533e024ad9c79ee6c428d`.
Use that commit as the primary package-matched upstream reference.
Use the `metalium-mesh` and `metalium-rope` trees as focused implementation references.
Use `rwkv-edit`, `rwkv-interp`, and `rwkv-probe` only for their distinct experimental work.
Do not use the fork's default `rknpu2-backend` branch as Metalium authority.
Use `ggml-org/llama.cpp` as the parent upstream for generic llama.cpp and ggml behavior.

### llama.cpp implementation references

Read these patches before you change RWKV-7 GPU decode, graph fusion, or linear-attention backend operators.

- [#26111: CUDA : add warp-per-row WKV7 kernel for single-token decode](https://github.com/ggml-org/llama.cpp/pull/26111)
  Patch: `.pi/references/rwkv-7/llama.cpp-prs/26111.patch`.
  Metadata: `.pi/references/rwkv-7/llama.cpp-prs/26111.json`.
- [#25601: vulkan backend ops: implemented GATED_LINEAR_ATTN](https://github.com/ggml-org/llama.cpp/pull/25601)
  Patch: `.pi/references/rwkv-7/llama.cpp-prs/25601.patch`.
  Metadata: `.pi/references/rwkv-7/llama.cpp-prs/25601.json`.
- [#25206: llama: optimize RWKV7 inference by fusing some graph operators](https://github.com/ggml-org/llama.cpp/pull/25206)
  Patch: `.pi/references/rwkv-7/llama.cpp-prs/25206.patch`.
  Metadata: `.pi/references/rwkv-7/llama.cpp-prs/25206.json`.
- [#18102: ggml-cuda: Delta-Net linear attention for Qwen3-Next](https://github.com/ggml-org/llama.cpp/pull/18102)
  Patch: `.pi/references/rwkv-7/llama.cpp-prs/18102.patch`.
  Metadata: `.pi/references/rwkv-7/llama.cpp-prs/18102.json`.

Treat these patches as implementation references.
Do not treat them as accepted upstream behavior.
Read each metadata file before you reuse its patch.
Run positive and negative tests for copied behavior.

## Qwen3.6-A3B model target references

- Read the Qwen3.6-35B-A3B bring-up tree at `.pi/references/qwen3-6-a3b/knight-ops-tt-metal`.
  Source: <https://github.com/Knight-Ops/tt-metal/tree/qwen3_6_a3b-bringup/models/demos/qwen3_6_a3b>.
  The sparse checkout uses branch `qwen3_6_a3b-bringup` and commit `37743f58d443be1eaf70ed10eb37b5377dc56b70`.
  The checkout contains only `models/demos/qwen3_6_a3b`.

Use this tree as the upstream bring-up reference for the Qwen3.6-A3B model target candidate.
Treat its PCC, performance, memory, and firmware observations as upstream claims until this repository reproduces them.
Record the Blackhole firmware floor (`>= 19.5.0`) and the pinned tt-metal `python_env` as machine facts, not as local evidence.
Do not run demo, server, or test commands from the reference tree without a bounded change.

- Read the Adartras independent tt-kernel vLLM bundle at `.pi/references/qwen3-6-a3b/adartras-tt-kernel`.
  Source: <https://huggingface.co/Adartras/qwen3.6-a3b-blackhole>.
  It is a code-only bundle (commit `42a71032659a9d0e3b860b6104c59cfa6ca82149`) serving the
  same model on one p150 with `ttnn >= 0.77`, BFP4 routed experts, and MTP spec-decode.
  This is a SECOND, independent Qwen3.6-A3B implementation; it is not the bring-up reference.
- Read the distilled transferable-technique study at `.pi/references/qwen3-6-a3b/LESSONS-FROM-ADARTRAS-BUNDLE.md`
  before changing the local `pkgs/qwen36` MTP, precision, or gated-delta code.
  It maps the bundle's distribution-exact speculative sampling, recurrent-state
  verify/commit contract, KV bf8 gating discipline, and trace-capture hygiene to the
  local 0.74 path, and it flags which lessons require the 0.77 pin and which do not.

Use the brought-up tree and the Adartras bundle as independent implementation references for the Qwen3.6-A3B target candidate.

## Adartras-bundle caveats (machine facts, not local evidence)

- The bundle is built against `ttnn 0.77.1.dev0+g9f9cd4fd590.d20260819` (2026-08-19 dev build); the local tt-metal pin is `0.74.0`.
- The bundle targets a single p150 (`1x1` mesh); the local serialized qwen36 service requires `P150x2`.
- Its routed-expert default is BFP4 because only that fits one 32 GB p150 (~17.5 GB vs ~35 GB BFP8).

The `.pi/` directory is not in version control.
Download a missing reference from its listed source before you use it.
