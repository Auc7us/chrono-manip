#!/usr/bin/env python3
"""Estimate the static tray and parked arm from both photographs; no dynamics edits."""
from pathlib import Path
import argparse
import json
import sys
import cv2
import numpy as np
from scipy.optimize import least_squares,differential_evolution

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from reference_camera import PHOTO_FIT_PROFILE,project_world,load_profile
from so101_ik import forward_kinematics
from scene_config import SHEET_SIZE,SHEET_CENTER


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wrist',type=Path,default=ROOT/'output/reference_inputs/reference_wrist.jpg')
    args=parser.parse_args();profile=load_profile(PHOTO_FIT_PROFILE)
    square=np.array([[-1,-1],[-1,1],[1,1],[1,-1]])*.025
    target=np.array([[220,255],[239,269],[224,285],[205,270]])
    def projected_tray(p):
        c,s=np.cos(p[2]),np.sin(p[2]);xy=square@np.array([[c,s],[-s,c]])+[p[0],p[1]]
        return project_world(np.column_stack([xy,np.full(4,.013)]),profile)
    best=None
    for reverse in [False,True]:
        for shift in range(4):
            observed=np.roll(target[::-1] if reverse else target,shift,axis=0)
            fit=least_squares(lambda p:(projected_tray(p)-observed).ravel(),[.08,.295,.7],
                              bounds=([-.05,.20,-np.pi],[.18,.40,np.pi]))
            if best is None or np.linalg.norm(fit.fun)<np.linalg.norm(best.fun):best=fit
    tray=best.x;cube=np.array([*tray[:2],.018])
    print('Tray:',tray,'corner RMS:',np.sqrt(np.mean(np.sum(best.fun.reshape(-1,2)**2,axis=1))),flush=True)
    assets=json.loads((ROOT/'assets/scene_assets.json').read_text())
    previous=json.loads((ROOT/'assets/reference_layout.json').read_text()) if (ROOT/'assets/reference_layout.json').exists() else None
    jaw_deg=previous['inspection_joint_deg'][5] if previous else 58.
    meshes={i:[] for i in range(7)}
    parts=assets['meshes']+[dict(link=5,file='wrist_camera_mount.obj')]
    for part in parts:
        path=ROOT/'assets/meshes'/part['file']
        vertices=np.array([[float(x) for x in line.split()[1:4]] for line in path.read_text().splitlines() if line.startswith('v ')])
        # Extremities retained through deterministic sampling; each body's convex
        # hull approximates its silhouette for the coarse pose search.
        meshes[part['link']].append(vertices[::max(1,len(vertices)//350)])
    meshes={i:np.concatenate(p) for i,p in meshes.items() if p}
    target_global=np.zeros((144,192),np.uint8)
    outline=np.array([[88,254],[84,241],[91,225],[82,216],[89,199],[121,177],[132,174],
        [132,168],[141,171],[146,180],[161,180],[174,167],[186,164],[195,172],[183,183],
        [173,188],[173,206],[161,217],[158,230],[143,245],[133,257],[111,261]])
    cv2.fillPoly(target_global,[np.rint(outline*.3).astype(np.int32)],1)
    wrist=cv2.imread(str(args.wrist));gray=cv2.cvtColor(wrist,cv2.COLOR_BGR2GRAY)
    _,labels=cv2.connectedComponents((gray<120).astype(np.uint8))
    target_wrist=cv2.resize((labels==labels[140,350]).astype(np.uint8),(160,120),interpolation=cv2.INTER_NEAREST)
    sx,sy,_=SHEET_SIZE;cy=SHEET_CENTER[1]
    corners=np.array([[-sx/2,cy-sy/2,.001],[-sx/2,cy+sy/2,.001],
                      [sx/2,cy+sy/2,.001],[sx/2,cy-sy/2,.001]])
    boundary=np.concatenate([a[None,:]*(1-np.linspace(0,1,24)[:,None])+b[None,:]*np.linspace(0,1,24)[:,None]
                             for a,b in zip(corners,np.roll(corners,-1,axis=0))])
    cam=assets['wrist_camera'];board=np.eye(4);board[:3,:3]=cam['rotation'];board[:3,3]=cam['board_center_m']
    optical=np.eye(4);optical[:3,3]=[cam['optical_offset_m'],0,0];optical[:3,:3]=np.diag([1.,-1.,-1.])
    local_camera=board@optical
    wrist_correction=ROOT/'assets/reference_wrist.json'
    if wrist_correction.exists():
        local_camera=local_camera@np.array(json.loads(wrist_correction.read_text())['optical_from_corrected_optical'])
    iou=lambda a,b:np.count_nonzero(a&b)/max(1,np.count_nonzero(a|b))
    def objective(q,details=False):
        frames=forward_kinematics(assets,[*q,jaw_deg])
        mask=np.zeros_like(target_global);penalty=0.
        for i,vertices in meshes.items():
            world=vertices@frames[i][:3,:3].T+frames[i][:3,3]
            penalty+=max(0,-world[:,2].min()-.002)*100
            uv=project_world(world,profile)*.3
            hull=cv2.convexHull(np.clip(uv,-1000,2000).astype(np.int32))
            cv2.fillConvexPoly(mask,hull,1)
        optical=frames[5]@local_camera
        wrist_profile={**profile,'optical_position_m':optical[:3,3].tolist(),
                       'world_from_optical_rotation':optical[:3,:3].tolist()}
        mat=np.zeros_like(target_wrist)
        uv=project_world(boundary,wrist_profile)/4
        cv2.fillPoly(mat,[np.clip(uv,-1000,2000).astype(np.int32)],1)
        cube_px=project_world(cube[None,:],wrist_profile)[0]
        # Silhouette plus the second camera's observed mat/cube constrain wrist
        # height and orientation much more strongly than a single arm silhouette.
        score=1.2*(1-iou(mask,target_global))+1.8*(1-iou(mat,target_wrist))
        score+=min(8.,np.sum(((cube_px-[590,268])/85)**2))+penalty
        if details:return dict(global_arm_iou=float(iou(mask,target_global)),wrist_mat_iou=float(iou(mat,target_wrist)),
                               wrist_cube_px=cube_px.tolist(),objective=float(score))
        return score
    bounds=[(15,140),(-10,140),(-155,5),(-100,150),(-180,180)]
    fit=differential_evolution(objective,bounds,popsize=9,maxiter=130,seed=19,tol=.003,polish=False,
                               x0=[60,35,-50,105,90],updating='immediate')
    report=dict(method='static layout fit to fixed and wrist photographs',
        tray_center_xy_m=tray[:2].tolist(),tray_yaw_rad=float(tray[2]),
        tray_corner_rms_px=float(np.sqrt(np.mean(np.sum(best.fun.reshape(-1,2)**2,axis=1)))),
        tray_coordinate_rms_px=float(np.sqrt(np.mean(best.fun**2))),
        cube_center_xy_m=tray[:2].tolist(),cube_yaw_rad=float(tray[2]),
        inspection_joint_deg=[*fit.x.tolist(),jaw_deg],pose_fit=objective(fit.x,True),
        assumptions=['Robot base retains the existing centered mounting transform',
                     'Approximate hand-traced global arm silhouette',
                     'Wrist and fixed camera share the fitted lens',
                     'Static inspection fit does not change automated home/physics'])
    (ROOT/'assets/reference_layout.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
