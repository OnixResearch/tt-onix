#!/usr/bin/env bash
# r[impl tenstorrent.training_platform.launch]
# Install the complete package-owned runtime closure without network access.
set -euo pipefail

WHEELHOUSE=/opt/tenstorrent-wheelhouse
REQUIREMENTS="$WHEELHOUSE/requirements.lock"
VENV=/tmp/venv

python3.12 -m venv "$VENV"
# shellcheck source=/dev/null
. "$VENV/bin/activate"

python -m pip install \
  --quiet \
  --no-cache-dir \
  --no-index \
  --no-deps \
  --find-links "$WHEELHOUSE" \
  --require-hashes \
  --requirement "$REQUIREMENTS"
python -m pip check

python /platform/smoke.py
