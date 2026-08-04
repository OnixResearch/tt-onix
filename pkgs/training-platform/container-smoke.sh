# r[impl tenstorrent.training_platform.launch]
# Runs inside the pinned TT-XLA base image. Installs the hash-pinned PJRT
# wheel and executes the bounded collective-free mesh smoke.
set -euo pipefail

PJRT_VERSION="@pjrt_version@"
PJRT_SHA256="@pjrt_sha256@"
JAX_VERSION="0.7.1"
JAXLIB_VERSION="0.7.1"
INDEX="https://pypi.eng.aws.tenstorrent.com"
WHEEL="pjrt_plugin_tt-${PJRT_VERSION}-cp312-cp312-manylinux_2_34_x86_64.whl"

python3.12 -m venv /tmp/venv
. /tmp/venv/bin/activate

python -m pip install --quiet --upgrade pip
python -m pip download --quiet --no-deps --dest /tmp/wheels \
  "pjrt-plugin-tt==${PJRT_VERSION}" --extra-index-url "$INDEX"

echo "${PJRT_SHA256}  /tmp/wheels/${WHEEL}" | sha256sum -c

python -m pip install --quiet "/tmp/wheels/${WHEEL}"
python -m pip install --quiet "jax==${JAX_VERSION}" "jaxlib==${JAXLIB_VERSION}"

python /platform/smoke.py
