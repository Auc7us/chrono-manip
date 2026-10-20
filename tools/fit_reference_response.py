#!/usr/bin/env python3
"""Fit fixed per-camera contrast/exposure response on corresponding flat regions."""
from pathlib import Path
import argparse
import json
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.special import expit,logit

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('renders',type=Path)
    parser.add_argument('--references',type=Path,default=ROOT/'output/reference_inputs')
    args=parser.parse_args();report={}
    for camera,reference in [('right_45','global'),('wrist','wrist')]:
        sim=cv2.imread(str(args.renders/(camera+'.png'))).astype(float)/255
        real=cv2.imread(str(args.references/('reference_'+reference+'.jpg'))).astype(float)/255
        edge=lambda a:cv2.magnitude(cv2.Sobel(a.mean(2).astype(np.float32),cv2.CV_32F,1,0),
                                   cv2.Sobel(a.mean(2).astype(np.float32),cv2.CV_32F,0,1))
        mask=((edge(sim)<.12)&(edge(real)<.12)).astype(np.uint8)
        mask=cv2.erode(mask,np.ones((5,5),np.uint8)).astype(bool)
        # Reject mismatched material/geometry patches; no image warp or texture
        # baking is used, only one monotonic exposure response for each camera.
        neutral=lambda a:(a.max(2)-a.min(2))<.13
        mask&=neutral(sim)&neutral(real)
        indices=np.flatnonzero(mask)[::3]
        x=logit(np.clip(sim.reshape(-1,3)[indices],.001,.999));y=real.reshape(-1,3)[indices]
        fit=least_squares(lambda p:(expit(p[0]*x+p[1])-y).ravel(),[1,0],
                          bounds=([.55,-2.2],[2.0,2.2]),loss='soft_l1',f_scale=.04,max_nfev=80)
        slope,bias=fit.x
        adjusted=expit(slope*logit(np.clip(sim,.000001,.999999))+bias)
        report[camera]=dict(contrast=float(slope),log_exposure_bias=float(bias),
            flat_pixels_used=int(len(indices)),raw_rgb_mae_8bit=float(np.abs(sim-real).mean()*255),
            adjusted_rgb_mae_8bit=float(np.abs(adjusted-real).mean()*255))
    result=dict(method='fixed monotonic camera response fitted to corresponding flat regions',
                response='sigmoid(contrast * logit(display_rgb) + log_exposure_bias)',
                cameras=report,assumptions=['Estimated USB camera ISP/exposure response, not measured shutter/ISO',
                                          'Same response applied to live feeds and recorded frames'])
    (ROOT/'assets/reference_response.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
