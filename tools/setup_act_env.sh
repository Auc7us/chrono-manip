#!/usr/bin/env bash
set -euo pipefail
act_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
act_venv="${act_root}/.venv-act"
export PIP_CACHE_DIR="${act_root}/output/act_setup/pip-cache"
mkdir -p "$PIP_CACHE_DIR"
if [ ! -x "${act_venv}/bin/python" ] || ! "${act_venv}/bin/python" -m pip --version >/dev/null 2>&1; then
    # The Chrono image omits Debian's ensurepip/venv package. Bootstrap the
    # environment locally instead of changing the container's system Python.
    act_bootstrap="${act_root}/output/act_setup/bootstrap"
    python3 -m pip install --target "$act_bootstrap" 'virtualenv>=20.26,<21'
    PYTHONPATH="${act_bootstrap}${PYTHONPATH:+:${PYTHONPATH}}" python3 -m virtualenv "$act_venv"
fi
"${act_venv}/bin/python" -m pip install --upgrade pip
"${act_venv}/bin/python" -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
"${act_venv}/bin/python" -m pip install lerobot==0.4.3 torchcodec==0.5 numpy==2.2.6 opencv-python-headless==4.11.0.86 scipy==1.15.3
"${act_venv}/bin/python" -m pip check
"${act_venv}/bin/python" - <<'PY'
import torch,torchvision,lerobot
print('Torch',torch.__version__,'torchvision',torchvision.__version__)
assert torch.cuda.is_available(),'CUDA is unavailable in the training environment'
print('GPU',torch.cuda.get_device_name(0))
print('BF16 support',torch.cuda.is_bf16_supported())
PY
