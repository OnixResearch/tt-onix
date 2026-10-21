#!/usr/bin/env bash
set -euo pipefail

repository_root=$(git rev-parse --show-toplevel)
if [ "$PWD" != "$repository_root" ]; then
  echo "tenstorrent-quality: run from the repository root" >&2
  exit 1
fi
# A path: flake would copy ignored runtime logs, models, and .pi/ into the Nix
# store. Evaluate the Git-tracked worktree instead; untracked files are not part
# of the release source and must be added to Git before they can be checked.
source="git+file://$repository_root"
nix flake check "$source" --all-systems --no-build -L
nix build "$source#checks.x86_64-linux.repository-quality" --no-link -L
