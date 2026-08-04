# r[impl tenstorrent.rwkv7_p150a_finetuning.compiler]
# Runs inside the pinned TT-XLA base image. Installs the host-verified
# wheels from the read-only /wheels mount and executes one probe.
# The host runner checks every wheel sha256 against the admitted manifest
# before this script starts, so the container performs no wheel downloads.
set -euo pipefail

python3.12 -m venv /tmp/venv
. /tmp/venv/bin/activate

python -m pip install --quiet --upgrade pip
python -m pip install --quiet /wheels/*.whl

exec python /probe/wkv7_probe.py "$@"
