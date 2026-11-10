#!/usr/bin/env bash
set -euo pipefail
act_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2
export HF_HOME="${act_root}/output/act_setup/hf-cache"
export TORCH_HOME="${act_root}/output/act_setup/torch-cache"
export HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=disabled
export ACCELERATE_MIXED_PRECISION=bf16
exec "${act_root}/.venv-act/bin/lerobot-train" "$@"
