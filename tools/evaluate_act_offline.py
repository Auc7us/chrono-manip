#!/usr/bin/env python3
"""Measure zero-latent ACT predictions on a separately exported held-out dataset."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from act_inference import load_act


def main():
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--samples',type=int,default=512);p.add_argument('--batch-size',type=int,default=4);a=p.parse_args()
    policy,pre,post=load_act(a.checkpoint);fps=json.loads((a.dataset/'meta/info.json').read_text())['fps']
    dataset=LeRobotDataset('local/so101-raised-validation',root=a.dataset,
        delta_timestamps={'action':[i/fps for i in range(policy.config.chunk_size)]},video_backend='pyav')
    indices=np.unique(np.linspace(0,len(dataset)-1,min(a.samples,len(dataset)),dtype=int))
    loader=DataLoader(Subset(dataset,indices),batch_size=a.batch_size,num_workers=2)
    absolute=np.zeros(6);squared=np.zeros(6);first=np.zeros(6);valid_count=0;rows=0;t=time.monotonic();phase_labels={};phase_errors={}
    metadata_path=a.dataset/'chrono_metadata.json'
    if metadata_path.exists():
        metadata=json.loads(metadata_path.read_text())
        for i,episode in enumerate(metadata['source_episodes']):
            path=ROOT/'output/pick_and_place'/episode['source_episode_path']/'trajectory.npz'
            if path.exists():
                with np.load(path) as saved:phase_labels[i]=saved['phase'].copy()
    with torch.inference_mode():
        for batch in loader:
            expected=batch['action'].clone();mask=~batch['action_is_pad']
            prediction=post(policy.predict_action_chunk(pre(batch))).cpu()
            difference=(prediction-expected).numpy();valid=mask.numpy()
            absolute+=(abs(difference)*valid[:,:,None]).sum((0,1));squared+=(difference**2*valid[:,:,None]).sum((0,1))
            first+=abs(difference[:,0]).sum(0);valid_count+=int(valid.sum());rows+=len(difference)
            # Phase labels are read only after prediction, for diagnostics.
            # They are never supplied to the policy or its VAE latent.
            for i,(episode,frame) in enumerate(zip(batch['episode_index'],batch['frame_index'])):
                if int(episode) not in phase_labels:continue
                phase=str(phase_labels[int(episode)][int(frame)])
                record=phase_errors.setdefault(phase,dict(count=0,total=np.zeros(6)))
                record['count']+=1;record['total']+=abs(difference[i,0])
    report=dict(checkpoint=str(a.checkpoint.resolve()),validation_dataset=str(a.dataset.resolve()),observations=rows,
        unpadded_future_targets=valid_count,chunk_size=policy.config.chunk_size,
        mean_absolute_error_per_joint=(absolute/valid_count).tolist(),root_mean_square_error_per_joint=np.sqrt(squared/valid_count).tolist(),
        first_action_mae_per_joint=(first/rows).tolist(),units=['degrees']*5+['percent'],elapsed_s=time.monotonic()-t,
        method='Inference prior latent zero; saved training normalization; no teacher-forced VAE latent from validation actions')
    report['first_action_mae_by_phase']={phase:dict(observations=value['count'],mae_per_joint=(value['total']/value['count']).tolist()) for phase,value in phase_errors.items()}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
