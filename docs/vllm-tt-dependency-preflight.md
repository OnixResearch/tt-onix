# vLLM TT dependency preflight

## Result

The current package set is not a drop-in environment for the reviewed official TT plugin. The Qwen runtime smoke passed, but it did not test vLLM.

The next build needs an isolated dependency profile. Do not replace dependencies in the deployed Qwen environment or suppress the new requirements to force installation.

## Sources

- Plugin commit: `bef89e429e202caa38d3e3c8a24da4e8cc02405a`.
- Plugin installer: `docs/install-vllm-tt.sh`.
- Plugin overrides: `docs/vllm-overrides.txt`.
- vLLM source: `https://github.com/vllm-project/vllm/archive/refs/tags/v0.26.0.tar.gz`.
- Unpacked source hash: `sha256-jFzV6vQX88FhemF98HmT5j3t6Trj5lXVlym4WD/X+Kw=`.

The vLLM source was fetched and inspected. Its `requirements/common.txt` and `pyproject.toml` supply the requirements below. The plugin overrides add further restrictions.

## Checked version differences

The base versions came from Nix evaluation of this repository's pinned package set.

| Package | Base version | Reviewed target requirement |
| --- | --- | --- |
| vLLM | 0.16.0 | 0.26.0, built with the empty target |
| Torch | 2.12.0 | vLLM build requirements pin 2.11.0 |
| Torchvision | 0.27.0 | Plugin installer pins 0.26.0 for its Torch pair |
| NumPy | 2.4.4 | Plugin install profile requires at least 1.24.4 and below 2 |
| Starlette | 0.52.1 | vLLM requires at least 1.0.1 |
| compressed-tensors | 0.15.0.1 | vLLM pins 0.17.0 |
| Transformers | 5.5.4 | Meets the plugin range: at least 5.5.3 and below 5.15.0 |
| FastAPI | 0.135.3 | Meets the vLLM range: at least 0.133.0 and below 0.137.0 |
| Pydantic | 2.12.5 | Meets the vLLM floor of 2.12.0 |
| depyf | 0.20.0 | Matches the vLLM pin |

This table is not a complete dependency resolution. It identifies concrete differences that a version-only override cannot resolve.

The Starlette floor is security-relevant: upstream cites CVE-2026-48710. Do not relax that floor. The plugin's NumPy and Torchvision restrictions describe its tested TT-Metal environment, not the already tested local Nix environment. A different profile needs its own evidence.

## Build and acceptance boundary

The plugin installer builds vLLM from source with `VLLM_TARGET_DEVICE=empty`. It installs common requirements explicitly to avoid CUDA wheel metadata. The PyPI CUDA wheel is not a substitute.

The plugin does not declare vLLM as a resolvable dependency. Its entry points are installed separately after the matching runtime. A plugin import alone therefore cannot establish a working vLLM stack.

A complete build must resolve the dependency profile, preserve a coherent Python/Torch/TT-NN ABI, pass the plugin's positive and negative tests, and then pass a bounded non-MTP model trial. The plugin's declared test dependency is pytest below 9.

No vLLM package build, plugin test suite, or vLLM serving run is claimed by this preflight. No dependency pin was relaxed. The fetched source and evaluated versions are preparation evidence only.

## Packages in this tree

Two pieces are now packaged independently of the vLLM core:

- `pkgs/vllm-tt-plugin` at rev `bef89e429e202caa38d3e3c8a24da4e8cc02405a`,
  version `0.1.0`. It builds with `tblib` only; `vllm` is intentionally absent.
  The build relaxes upstream's setuptools `<81` build cap because nixpkgs ships
  setuptools 83. `pythonImportsCheck` imports the package root; the plugin's
  platform entry point still requires a matching empty-target vLLM.
- `pkgs/faster-fifo` 1.5.2 for the plugin's multiprocessing transport.

Neither package resolves the dependency conflict in the table above, and no
vLLM core package or serving run is claimed.

Local evidence under `~/.cache/tt-upgrade-review/evidence/`:

- `vllm026-prefetch.json`.
- `vllm-base-versions.json`.
