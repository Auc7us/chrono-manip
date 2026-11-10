#!/usr/bin/env python3
"""Run lerobot-train with Gaussian noise added to the normalized observation.state during training only.

A policy that copies its joint input keeps whatever position error it already has. Noise of a few
degrees makes that input unreliable, so the cameras must carry the cube position. Inference is unchanged.
Usage: train_act_state_noise.py --state-noise-std 0.1 [lerobot-train arguments]
"""
import sys
import torch
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.scripts.lerobot_train import main

STATE='observation.state'


def patch(std):
    forward=ACTPolicy.forward
    def noisy_forward(self,batch,*args,**kwargs):
        if self.training and STATE in batch:
            batch=dict(batch);batch[STATE]=batch[STATE]+std*torch.randn_like(batch[STATE])
        return forward(self,batch,*args,**kwargs)
    ACTPolicy.forward=noisy_forward


if __name__=='__main__':
    if len(sys.argv)<3 or sys.argv[1]!='--state-noise-std':sys.exit(__doc__)
    std=float(sys.argv[2])
    if not std>=0:sys.exit('--state-noise-std must be nonnegative')
    patch(std);sys.argv=[sys.argv[0],*sys.argv[3:]];print(f'Training-time state noise: {std} (normalized units)',flush=True)
    sys.exit(main())
