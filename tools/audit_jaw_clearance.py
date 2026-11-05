#!/usr/bin/env python3
"""Measure exact jaw CAD extrema along every observed reference arm pose."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from scipy.spatial import ConvexHull

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.fit_metric_twin import axis_rotations


def audit(assets,mapping,data):
    q=np.deg2rad(data['observation_state']*mapping['scale']+mapping['offset_deg'])
    pivots=np.array([j['pivot_m'] for j in assets['joints']]);axes=np.array([j['axis'] for j in assets['joints']]);axes/=np.linalg.norm(axes,axis=1)[:,None]
    frame=np.broadcast_to(np.array(assets['base_world_from_cad']),(len(q),4,4)).copy()
    rotations=axis_rotations(axes,q);heights={}
    for j in range(6):
        t=np.broadcast_to(np.eye(4),(len(q),4,4)).copy();t[:,:3,:3]=rotations[:,j];t[:,:3,3]=pivots[j]-np.einsum('nij,j->ni',rotations[:,j],pivots[j]);frame=frame@t
        if j not in [4,5]:continue
        name='fixed_jaw' if j==4 else 'moving_jaw'
        vertices=np.array([[float(x) for x in line.split()[1:4]] for line in (ROOT/'assets/meshes'/f'{name}_collision.obj').read_text().splitlines() if line.startswith('v ')])
        vertices=vertices[ConvexHull(vertices).vertices]
        heights[name]=np.min(np.einsum('ni,vi->nv',frame[:,2,:3],vertices)+frame[:,2,3,None],axis=1)
    episodes=[]
    for ep in np.unique(data['episode_index']):
        selection=data['episode_index']==ep;row=dict(episode=int(ep),held_out=bool(ep>=8))
        for name,z in heights.items():
            indices=np.flatnonzero(selection);index=indices[np.argmin(z[selection])]
            row[name]=dict(minimum_world_z_m=float(z[index]),penetration_below_mat_m=max(0.,.001-float(z[index])),
                frames_below_mat=int(np.sum(z[selection]<.001)),minimum_dataset_index=int(index),minimum_timestamp_s=float(data['timestamp'][index]))
        episodes.append(row)
    return dict(method='Exact extrema of full jaw meshes at recorded measured joint states; no physics or object correction',
        mat_plane_world_z_m=.001,joint_map=mapping,episodes=episodes)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--fit',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assets=json.loads((ROOT/'assets/scene_assets.json').read_text());mapping=json.loads((ROOT/'assets/video_joint_map.json').read_text())
    if a.fit:
        fit=json.loads(a.fit.read_text());mapping=dict(scale=fit['scale'],offset_deg=fit['offset_deg'])
        base=np.eye(4);base[:3,:3]=np.diag([-1.,-1.,1.]);base[:3,3]=[0.,.054,-.00039]+np.array(fit['base_translation_delta_m']);assets['base_world_from_cad']=base.tolist()
    with np.load(ROOT/'output/video_fit/reference_trajectory.npz') as data:report=audit(assets,mapping,data)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['episodes'],indent=2))


if __name__=='__main__':main()
