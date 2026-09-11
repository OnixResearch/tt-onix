## Why

Every measured path to continuous batching on this hardware runs through a
serving stack this repository does not yet have. A third party serving
Qwen3.8-27B on two Blackhole p150a cards reaches 100.5 aggregate tokens/second
at B=8 through `vllm-tt-plugin`, and the repository's own serialized service
admits only B=4.

The preflight already established that the current package set is not a drop-in
environment for the reviewed plugin. The differences are concrete:

| Package | Base version | Reviewed target requirement |
| --- | --- | --- |
| vLLM | 0.16.0 | 0.26.0, built with the empty target |
| Torch | 2.12.0 | 2.11.0 |
| Torchvision | 0.27.0 | 0.26.0 |
| NumPy | 2.4.4 | at least 1.24.4 and below 2 |
| Starlette | 0.52.1 | at least 1.0.1 |
| compressed-tensors | 0.15.0.1 | 0.17.0 |

A version-only override cannot resolve this set. The Starlette floor is
security-relevant: upstream cites CVE-2026-48710, so the floor must not be
relaxed. The NumPy and Torchvision restrictions describe the plugin's tested
TT-Metal environment, not this repository's tested Nix environment.

The preflight also recorded that the plugin does not declare vLLM as a
resolvable dependency, so a plugin import alone cannot establish a working
stack. No vLLM serving run is claimed today.

## What Changes

- Build a vLLM package from source with `VLLM_TARGET_DEVICE=empty` against a
  pinned vLLM tag, installed without CUDA wheel metadata.
- Add an isolated dependency profile that carries the plugin's requirements
  without changing the deployed Qwen environment.
- Preserve the Starlette security floor and record it as a non-relaxable
  constraint.
- Prove a coherent Python, Torch, and TT-NN ABI before any serving run.
- Record the plugin's declared test dependency and run the plugin's own
  positive and negative tests.
- Run one bounded non-MTP model trial through the stack and record the result.

This change produces a serving profile and its evidence. It does not replace
the serialized service, does not admit a serving route, and does not claim a
throughput result.

## Impact

- **Files**: a new `pkgs/` vLLM profile, the Qwen package's optional serving
  entry point, this change directory, and the evidence tree.
- **Testing**: profile resolution, ABI coherence checks, plugin positive and
  negative tests, package build, and one bounded non-MTP model trial.
- **Production**: unchanged. The serialized P150x2 service remains the served
  route and no dependency in the deployed environment changes.
