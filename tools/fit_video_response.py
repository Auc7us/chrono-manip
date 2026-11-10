#!/usr/bin/env python3
"""Fit one fixed response per camera across raw, synchronized replay samples."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.special import expit,logit

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('renders',type=Path)
    p.add_argument('--output',type=Path,default=ROOT/'assets/video_response.json')
    a=p.parse_args();report={}
    for camera in ['external','wrist']:
        inputs=[];targets=[];pairs=[]
        for source in sorted(a.renders.glob(camera+'_sim_*.png')):
            target=source.with_name(source.name.replace('_sim_','_real_'))
            sim=cv2.imread(str(source)).astype(float)/255;real=cv2.imread(str(target)).astype(float)/255
            edge=lambda im:cv2.magnitude(cv2.Sobel(im.mean(2).astype(np.float32),cv2.CV_32F,1,0),cv2.Sobel(im.mean(2).astype(np.float32),cv2.CV_32F,0,1))
            mask=cv2.erode(((edge(sim)<.10)&(edge(real)<.10)).astype(np.uint8),np.ones((7,7),np.uint8)).astype(bool)
            mask&=((sim.max(2)-sim.min(2))<.12)&((real.max(2)-real.min(2))<.12)
            # Reject silhouette/material disagreements while preserving both
            # white panels and dark mat in the response fit.
            mask&=((sim.mean(2)>.35)==(real.mean(2)>.35))
            indices=np.flatnonzero(mask)[::8]
            inputs.append(logit(np.clip(sim.reshape(-1,3)[indices],.001,.999)))
            targets.append(real.reshape(-1,3)[indices]);pairs.append((sim,real))
        if not inputs:raise ValueError('No raw replay sample pairs found')
        x=np.concatenate(inputs);y=np.concatenate(targets)
        fit=least_squares(lambda p:(expit(p[0]*x+p[1])-y).ravel(),[1.,.5],
            bounds=([.4,-2.5],[2.5,3.]),loss='soft_l1',f_scale=.04,max_nfev=80)
        slope,bias=fit.x
        report['right_45' if camera=='external' else camera]=dict(contrast=float(slope),
            log_exposure_bias=float(bias),sample_frames=len(pairs),flat_rgb_pixels_used=len(x),
            raw_rgb_mae_8bit=float(np.mean([np.abs(s-r).mean()*255 for s,r in pairs])),
            adjusted_rgb_mae_8bit=float(np.mean([np.abs(expit(slope*logit(np.clip(s,.000001,.999999))+bias)-r).mean()*255 for s,r in pairs])))
    result=dict(source='green_cube_red_plate_1 episode 0, raw Chrono replay samples at 3 s intervals',
        method='one fixed monotonic response per camera, fitted jointly across multiple wrist poses',
        response='sigmoid(contrast * logit(display_rgb) + log_exposure_bias)',cameras=report,
        assumptions=['No per-frame color fitting or exposure changes; same filter used in live views and datasets',
            'Fit pixels reject edges and mismatched materials; reported RGB MAE includes the complete frame',
            'Photographic response estimate; not measured shutter, ISO or spectral sensitivity'])
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
