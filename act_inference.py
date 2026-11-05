"""Load standard LeRobot ACT weights and their saved normalization pipelines."""
import math
from pathlib import Path
import os

ROOT=Path(__file__).resolve().parent
os.environ.setdefault('HF_HOME',str(ROOT/'output/act_setup/hf-cache'))
os.environ.setdefault('TORCH_HOME',str(ROOT/'output/act_setup/torch-cache'))
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY','1')
import torch
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_pre_post_processors


def load_act(checkpoint,device='cuda',action_steps=None,temporal_ensemble_coeff=None):
    checkpoint=str(Path(checkpoint).resolve())
    config=PreTrainedConfig.from_pretrained(checkpoint,local_files_only=True)
    if action_steps is not None:
        if not 1<=action_steps<=config.chunk_size:raise ValueError('Action steps must lie between 1 and chunk size')
        config.n_action_steps=action_steps
    if temporal_ensemble_coeff is not None:
        # LeRobot weights the oldest prediction as exp(0); a negative coefficient favors the newest.
        if not math.isfinite(temporal_ensemble_coeff):raise ValueError('Temporal ensemble coefficient must be finite')
        config.temporal_ensemble_coeff=temporal_ensemble_coeff;config.n_action_steps=1
    policy=ACTPolicy.from_pretrained(checkpoint,config=config,local_files_only=True)
    policy.to(device).eval();policy.reset()
    pre,post=make_pre_post_processors(policy.config,pretrained_path=checkpoint,
        preprocessor_overrides={'device_processor':{'device':device}})
    return policy,pre,post
