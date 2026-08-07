#!/usr/bin/env bash
# r[impl tenstorrent.rwkv7_p150a_finetuning.compiler]
# Runs inside the pinned TT-XLA base image. Installs the host-verified
# runtime wheels from the read-only /wheels mount, then installs their
# declared dependency set from the package indexes.
#
# The pjrt-plugin-tt wheel declares torch and torch-xla as URL
# dependencies. Installing the local wheel files and the URL requirements
# in one resolver pass conflicts, so the runtime wheels are installed with
# --no-deps and the dependency set is installed explicitly afterwards.
# `pip check` gates the declared dependency graph before any probe runs.
# The exact leaf versions are recorded in the probe evidence by the host
# runner through `pip freeze`.
set -euo pipefail

# Identity-critical wheels come from host-verified local bytes; jax/jaxlib
# keep the admitted platform manifest versions. All other dependencies are
# installed with the exact pins declared in the wheel metadata.
PINNED_VERSION_JAX="0.7.1"
PINNED_VERSION_JAXLIB="0.7.1"
TT_INDEX="https://pypi.eng.aws.tenstorrent.com"

python3.12 -m venv /tmp/venv
. /tmp/venv/bin/activate

python -m pip install --quiet --upgrade pip

# Manifest-verified runtime wheels: host checked every sha256 before mount.
# They install with --no-deps so the exact bytes are used.
python -m pip install --quiet --no-deps /wheels/*.whl

# The manifest pins the jax/jaxlib pair exactly.
python -m pip install --quiet \
  "jax==${PINNED_VERSION_JAX}" \
  "jaxlib==${PINNED_VERSION_JAXLIB}"

# Install the exact pinned dependency closure parsed from wheel metadata.
# Direct-URL requirements are skipped because those identity wheels are
# already installed from local bytes.
python /probe/install_deps.py > /tmp/deps.txt
if [ -s /tmp/deps.txt ]; then
  # shellcheck disable=SC2046
  python -m pip install --quiet $(cat /tmp/deps.txt) --extra-index-url "$TT_INDEX"
fi

python -m pip check
python -m pip freeze > /evidence/pip-freeze.txt

exec python /probe/wkv7_probe.py "$@"
