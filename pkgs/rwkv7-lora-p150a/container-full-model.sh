# r[impl tenstorrent.rwkv7_p150a_finetuning.graph]
# Runs inside the pinned TT-XLA base image for the full-model graph. Same
# wheel-install policy as container-probe.sh, plus safetensors to read the
# pinned checkpoint.
set -euo pipefail

PINNED_VERSION_JAX="0.7.1"
PINNED_VERSION_JAXLIB="0.7.1"
TT_INDEX="https://pypi.eng.aws.tenstorrent.com"

python3.12 -m venv /tmp/venv
. /tmp/venv/bin/activate

python -m pip install --quiet --upgrade pip
python -m pip install --quiet --no-deps /wheels/*.whl
python -m pip install --quiet \
  "jax==${PINNED_VERSION_JAX}" \
  "jaxlib==${PINNED_VERSION_JAXLIB}"

python /probe/install_deps.py > /tmp/deps.txt
if [ -s /tmp/deps.txt ]; then
  # shellcheck disable=SC2046
  python -m pip install --quiet $(cat /tmp/deps.txt) --extra-index-url "$TT_INDEX"
fi
python -m pip install --quiet safetensors --extra-index-url "$TT_INDEX"

python -m pip check
mkdir -p /evidence
python -m pip freeze > /evidence/pip-freeze.txt

exec python /probe/full_model_probe.py "$@"
