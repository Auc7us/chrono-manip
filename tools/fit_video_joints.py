#!/usr/bin/env python3
"""Estimate recorded motor-to-CAD coordinates from multiple wrist mat views.

This visual alignment is for replay, not a hardware motor calibration.
"""
import argparse
import json
import sys
import itertools
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.ndimage import distance_transform_edt,map_coordinates

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from so101_ik import forward_kinematics
from reference_camera import load_profile,project_world


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('trajectory',type=Path)
    p.add_argument('--output',type=Path,default=ROOT/'assets/video_joint_map.json')
    p.add_argument('--samples',type=Path,help='optional replay samples for fitting visible mat boundaries across more poses')
    args=p.parse_args()
    data=np.load(args.trajectory);states=data['observation_state']
    assets=json.loads((ROOT/'assets/scene_assets.json').read_text());cam=assets['wrist_camera']
    board=np.eye(4);board[:3,:3]=cam['rotation'];board[:3,3]=cam['board_center_m']
    optical=np.diag([1.,-1.,-1.,1.]);optical[0,3]=cam['optical_offset_m']
    local=board@optical@np.array(json.loads((ROOT/'assets/reference_wrist.json').read_text())['optical_from_corrected_optical'])
    profile=load_profile(ROOT/'assets/video_camera.json')
    world=np.array([[-.2286,.1524,.001],[-.2286,.4572,.001],
                    [.2286,.4572,.001],[.2286,.1524,.001]])
    observations=[(0,[[48,344],[187,114],[526,207],[537,463]]),
        (90,[[77,340],[189,44],[568,222],[None,None]]),
        (180,[[92,407],[77,53],[629,220],[500,454]]),
        (840,[[104,348],[None,None],[619,92],[560,377]])]
    fields=[]
    boundary=np.concatenate([a+(b-a)*np.linspace(0,1,48)[:,None]
        for a,b in zip(world,np.roll(world,-1,axis=0))])
    if args.samples:
        import cv2
        records=json.loads((args.samples/'replay.json').read_text())['records']
        for file in sorted(args.samples.glob('wrist_real_*.png')):
            index=records[int(file.stem.rsplit('_',1)[1])]['dataset_index']
            gray=cv2.imread(str(file),cv2.IMREAD_GRAYSCALE)
            n,labels,stats,_=cv2.connectedComponentsWithStats((gray<120).astype(np.uint8))
            # The mat spans the central image; exclude dim room components.
            candidates=[i for i in range(1,n) if stats[i,cv2.CC_STAT_AREA]>15000 and
                np.count_nonzero(labels[70:270,150:500]==i)>5000]
            if not candidates:continue
            label=max(candidates,key=lambda i:stats[i,cv2.CC_STAT_AREA])
            mask=labels==label
            field=distance_transform_edt(~mask)-distance_transform_edt(mask)
            fields.append((index,field))
    initial=np.r_[[.9]*5,[0,90,-90,0,70]]
    def predictions(p,index,points=world):
        q=np.r_[states[index,:5]*p[:5]+p[5:],80.]
        pose=forward_kinematics(assets,q)[5]@local
        return project_world(points,{**profile,'optical_position_m':pose[:3,3],
                                   'world_from_optical_rotation':pose[:3,:3]})
    def residual(p):
        error=[]
        for index,pixels in observations:
            pixels=np.array(pixels,dtype=float);valid=np.isfinite(pixels[:,0])
            error.extend(((predictions(p,index)-pixels)[valid]/3).ravel())
        for index,field in fields:
            uv=predictions(p,index,boundary)
            valid=(uv[:,0]>8)&(uv[:,0]<632)&(uv[:,1]>8)&(uv[:,1]<472)
            valid&=~((uv[:,0]>160)&(uv[:,0]<500)&(uv[:,1]>270))
            distance=map_coordinates(field,[uv[:,1],uv[:,0]],order=1,mode='nearest')
            error.extend(np.clip(distance,-60,60)*valid/8)
        return np.r_[error,(np.abs(p[:5])-.9)/.20,(p[5:]-initial[5:])/25]
    fit=None
    for pan_sign,roll_sign in itertools.product([-1,1],repeat=2):
        signs=np.array([pan_sign,1,1,1,roll_sign])
        start=initial.copy();start[:5]*=signs
        low=np.where(signs>0,.2,-1.4);high=np.where(signs>0,1.4,-.2)
        candidate=least_squares(residual,start,x_scale='jac',max_nfev=1200,
            bounds=(np.r_[low,[-35,55,-125,-35,35]],np.r_[high,[35,125,-55,35,110]]))
        if fit is None or np.linalg.norm(candidate.fun)<np.linalg.norm(fit.fun):fit=candidate
    result=dict(method='multi-pose wrist mat-corner alignment; approximate visual motor map',
        source='green_cube_red_plate_1 observation.state; video frames 0, 90, 180, 840',
        equation='cad_degrees = recorded_state * scale + offset_deg',
        scale=[*fit.x[:5].tolist(),-1.],offset_deg=[*fit.x[5:].tolist(),92.],
        units='recorded normalized motor coordinates to degrees offset from CAD assembly',
        corner_coordinate_rms_px=float(np.sqrt(np.mean(np.concatenate([
            (predictions(fit.x,i)-np.array(pts,dtype=float))[np.isfinite(np.array(pts,dtype=float)[:,0])].ravel()
            for i,pts in observations])**2))),boundary_sample_frames=len(fields),
        measurements=[dict(chunk_frame=i,reference_px=pts,projected_px=predictions(fit.x,i).tolist()) for i,pts in observations],
        assumptions=['Mat corners hand-located and some partly occluded; wrist extrinsic estimate retained',
            'Gripper scale/offset estimated from visible jaw opening; no measured servo calibration',
            'Visual replay only; does not establish contact/dynamic or hardware control accuracy'])
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
