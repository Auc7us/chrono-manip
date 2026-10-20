#!/usr/bin/env python3
"""Fit the fixed camera to measured 2D box/mat landmarks (SciPy required).

This estimates a pose and lens from one image. It is not a checkerboard camera
calibration. Pixel landmarks, priors and residuals are retained for review.
"""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
from scipy.optimize import least_squares

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scene_config import WIDTH,DEPTH,HEIGHT,SHEET_SIZE,SHEET_CENTER


def rotation(yaw,pitch,roll):
    yaw,pitch,roll=np.deg2rad([yaw,pitch,roll])
    forward=np.array([-np.cos(pitch)*np.cos(yaw),np.cos(pitch)*np.sin(yaw),-np.sin(pitch)])
    left=np.cross([0,0,1],forward);left/=np.linalg.norm(left)
    up=np.cross(forward,left)
    return np.column_stack([forward,left*np.cos(roll)+up*np.sin(roll),
                            up*np.cos(roll)-left*np.sin(roll)])


def project(points,p):
    # p = optical position XYZ, yaw/pitch/roll, focal pixels, cx/cy, k1/k2.
    local=(points-p[:3])@rotation(*p[3:6])
    xy=-local[:,1:3]
    radius=np.linalg.norm(xy,axis=1)
    theta=np.arctan2(radius,local[:,0])
    radial=theta*(1+p[9]*theta**2+p[10]*theta**4)
    return p[6]*xy/np.maximum(radius[:,None],1e-12)*radial[:,None]+p[7:9]


def landmarks():
    w,d=WIDTH,DEPTH;sx,sy,_=SHEET_SIZE;cy=SHEET_CENTER[1]
    return dict(
        floor_far_front=([-w/2,0,0],[176,142]),
        floor_far_rear=([-w/2,d,0],[347,156]),
        floor_near_front=([w/2,0,0],[50,416]),
        floor_near_rear=([w/2,d,0],[407,451]),
        mat_far_front=([-sx/2,cy-sy/2,.001],[191,182]),
        mat_far_rear=([-sx/2,cy+sy/2,.001],[302,193]),
        mat_near_front=([sx/2,cy-sy/2,.001],[143,335]),
        mat_near_rear=([sx/2,cy+sy/2,.001],[306,350]))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'assets/reference_camera.json')
    args=parser.parse_args()
    marks=landmarks();world=np.array([v[0] for v in marks.values()]);pixels=np.array([v[1] for v in marks.values()])
    initial=np.array([WIDTH/2-.030,DEPTH/2,.480,15,45,0,250,320,240,0,0.])
    lower=[WIDTH/2-.10,.16,.41,-25,20,-30,150,280,200,-.30,-.10]
    upper=[WIDTH/2+.03,.46,.54,55,75,30,450,360,280,.30,.15]
    def residual(p):
        measured=(project(world,p)-pixels).ravel()
        # Weak construction priors resolve pose/lens ambiguities from a single view.
        prior=np.array([(p[0]-(WIDTH/2-.03))/.03,(p[2]-.48)/.03,
                        (p[7]-320)/24,(p[8]-240)/24,p[9]/.15,p[10]/.08])
        return np.r_[measured,prior]
    best=None
    for yaw in [0,15,30]:
        initial[3]=yaw
        fit=least_squares(residual,initial,bounds=(lower,upper),x_scale='jac',max_nfev=3000,
                          ftol=1e-12,xtol=1e-12,gtol=1e-10)
        if best is None or np.linalg.norm(fit.fun)<np.linalg.norm(best.fun):best=fit
    p=best.x;pred=project(world,p);errors=np.linalg.norm(pred-pixels,axis=1)
    report=dict(source='2026-10-19-161325.jpg',method='single-image box/mat landmark fit with construction priors',
        pixel_origin='top-left',image_size=[640,480],
        optical_position_m=p[:3].tolist(),yaw_pitch_roll_deg=p[3:6].tolist(),
        world_from_optical_rotation=rotation(*p[3:6]).tolist(),
        lens=dict(model='equidistant_polynomial',focal_px=float(p[6]),principal_point_px=p[7:9].tolist(),
                  theta_coefficients=p[9:11].tolist()),
        landmark_rms_px=float(np.sqrt(np.mean(errors**2))),
        landmarks={k:dict(world_m=world[i].tolist(),reference_px=pixels[i].tolist(),
                          projected_px=pred[i].tolist(),error_px=float(errors[i])) for i,k in enumerate(marks)},
        uncertainties=['Landmarks hand-located to approximately 1–3 px',
            '35 x 24 x 20 inch box and centered 18 x 12 inch mat assumed',
            'A single image does not establish an exact unique physical calibration'])
    args.output.parent.mkdir(exist_ok=True,parents=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['landmarks','uncertainties']},indent=2))
    for name,e in zip(marks,errors):print(name,round(e,3))


if __name__=='__main__':main()
