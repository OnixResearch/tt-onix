#!/usr/bin/env bash
set -euo pipefail

repository_root=$(git rev-parse --show-toplevel)
if [ "$PWD" != "$repository_root" ]; then
  echo "tenstorrent-quality: run from the repository root" >&2
  exit 1
fi
if [ -d generated ]; then
  echo "tenstorrent-quality: generated/ contains private runtime output" >&2
  exit 1
fi

nix flake check "path:$repository_root" --all-systems --no-build -L
nix build "path:$repository_root#checks.x86_64-linux.repository-quality" --no-link -L
