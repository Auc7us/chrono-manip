#!/usr/bin/env bash
# Run inside chrono-orb, from any working directory. Extra CLI flags override defaults.
set -euo pipefail
act_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [ ! -f "${act_root}/output/act_data/train/chrono_metadata.json" ]; then
    echo 'The full training export has not completed.' >&2
    exit 1
fi
exec "${act_root}/train-act.sh" \
    --dataset.repo_id=local/so101-raised-train \
    --dataset.root="${act_root}/output/act_data/train" \
    --dataset.video_backend=pyav \
    --policy.type=act --policy.device=cuda --policy.push_to_hub=false \
    --policy.chunk_size=30 --policy.n_action_steps=5 \
    --policy.optimizer_lr=0.0001 --policy.optimizer_lr_backbone=0.00001 \
    --batch_size=8 --steps=20000 --num_workers=4 \
    --eval_freq=0 --log_freq=100 --save_freq=2000 \
    --wandb.enable=false --seed=2026 \
    --output_dir="${act_root}/output/act_runs/raised_act_001" "$@"
