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
# Bootstrap the matching CLI locally: an unpatched daemon cannot schedule its
# own repair through a builder URI that exceeds the lock filesystem's NAME_MAX.
# Only this bootstrap disables builders; the quality build retains the configured
# machines and plugin-files, and passes the documented build-hook argv vector.
hook_package=$(nix --option plugin-files '' build "$source#nix-patched-build-remote" \
  --builders '' --max-jobs 1 --no-link --print-out-paths -L)
hook="$hook_package/bin/nix __build-remote"
"$hook_package/bin/nix" --option build-hook "$hook" flake check "$source" --all-systems --no-build -L
"$hook_package/bin/nix" --option build-hook "$hook" build "$source#checks.x86_64-linux.repository-quality" --no-link -L
