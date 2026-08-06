#!/usr/bin/env bash
# r[impl tenstorrent.rwkv7_p150a_finetuning.graph]
# Install only the package-owned, hash-locked offline wheel closure.
set -euo pipefail

WHEELHOUSE=/wheels
VENV=/tmp/venv

python3.12 -m venv "$VENV"
# shellcheck source=/dev/null
. "$VENV/bin/activate"
python -m pip install --quiet --no-cache-dir --no-index --no-deps \
  --find-links "$WHEELHOUSE" --require-hashes \
  --requirement "$WHEELHOUSE/requirements.lock"
python -m pip check
mkdir -p /evidence
python -m pip freeze > /evidence/pip-freeze.txt

exec python /probe/full_model_probe.py "$@"
