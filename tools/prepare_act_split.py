#!/usr/bin/env python3
"""Split successful demonstrations by complete tray-layout groups before export."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def usable(episode,source=None):
    """Complete demos that place the cube flat in the tray. Jaw-into-cube penetration past 0.5 mm is
    tolerated (tilted far grasps squeeze harder); early jaw contact, environment penetration and wrist
    tracking limits still apply."""
    if not (episode['complete'] and episode['frames']):return False
    if episode['success']:return True
    if source is None:return False
    e=json.loads((Path(source)/episode['path']/'episode.json').read_text())
    return bool(e['grasped'] and e['lifted'] and e['released'] and e['cube_in_tray'] and not e['premature_jaw_contact']
        and e['max_environment_penetration_m']<.0005 and e['max_tcp_tracking_error_m']<.0005)


def split_episodes(manifest,validation_fraction=.1,seed=2026,source=None):
    interval=manifest['tray_every']
    if interval<1:raise ValueError('Grouped split requires randomized tray layouts (--tray-every > 0)')
    groups={}
    for episode in manifest['episodes']:
        if not usable(episode,source):continue
        key=(episode.get('worker_id',0),episode.get('worker_episode_index',episode['index'])//interval)
        groups.setdefault(key,[]).append(episode)
    keys=sorted(groups)
    if len(keys)<2:raise ValueError('Need at least two successful tray-layout groups')
    rng=np.random.default_rng(seed);order=rng.permutation(len(keys));n=max(1,min(len(keys)-1,round(len(keys)*validation_fraction)))
    held_out={keys[i] for i in order[:n]}
    train=sorted(e['index'] for key,episodes in groups.items() if key not in held_out for e in episodes)
    validation=sorted(e['index'] for key,episodes in groups.items() if key in held_out for e in episodes)
    return dict(train_indices=train,validation_indices=validation,training_groups=[list(k) for k in keys if k not in held_out],
        validation_groups=[list(k) for k in keys if k in held_out],seed=seed,validation_fraction=validation_fraction,
        rule='Complete episodes placing the cube flat in the tray (jaw penetration not limited); all repeated tray layouts within a worker stay in the same partition')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--validation-fraction',type=float,default=.1);p.add_argument('--seed',type=int,default=2026);a=p.parse_args()
    if not 0<a.validation_fraction<1:p.error('validation fraction must be between 0 and 1')
    if a.output.exists():raise FileExistsError(a.output)
    raw=(a.source/'dataset.json').read_bytes();manifest=json.loads(raw)
    report=split_episodes(manifest,a.validation_fraction,a.seed,a.source)
    report.update(source_dataset=str(a.source.resolve()),source_manifest_sha256=hashlib.sha256(raw).hexdigest(),fps=manifest['fps'],
        successful_episodes=len(report['train_indices'])+len(report['validation_indices']),total_attempts=len(manifest['episodes']),
        tray_geometry=manifest.get('tray_geometry'))
    # Verify the grouping schedule against the actual recorded tray poses.
    poses={}
    for episode in manifest['episodes']:
        key=(episode.get('worker_id',0),episode.get('worker_episode_index',episode['index'])//manifest['tray_every'])
        info=json.loads((a.source/episode['path']/'episode.json').read_text());pose=np.r_[info['tray_xy_m'],info['tray_yaw_rad']]
        if key in poses:np.testing.assert_allclose(pose,poses[key],atol=1e-10,rtol=0)
        poses[key]=pose
    a.output.mkdir(parents=True)
    for name in ['train','validation']:(a.output/(name+'.json')).write_text(json.dumps(report[name+'_indices'])+'\n')
    (a.output/'split.json').write_text(json.dumps(report,indent=2)+'\n')
    print(f'{len(report["train_indices"])} training / {len(report["validation_indices"])} validation successful episodes; no shared tray groups.')


if __name__=='__main__':main()
