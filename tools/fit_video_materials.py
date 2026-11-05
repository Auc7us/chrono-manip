#!/usr/bin/env python3
"""Estimate fixed colored PLA reflectances from the aligned initial views."""
import json
import sys
from pathlib import Path
import cv2
import numpy as np
from scipy.special import expit,logit

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from reference_objects import object_mask


def main():
    response=json.loads((ROOT/'assets/video_response.json').read_text())['cameras'];measurements={k:[] for k in ['pla','red']}
    raw_metadata=json.loads((ROOT/'output/metric_refit/raw_initial/object_and_tray.json').read_text())
    base_colors=raw_metadata['appearance']['materials']['linear_rgb_reflectance']
    for camera,sim_name in [('wrist','wrist'),('external','right_45')]:
        real=cv2.cvtColor(cv2.imread(str(ROOT/f'output/video_fit/real_{camera}_000.png')),cv2.COLOR_BGR2RGB)/255
        sim=cv2.cvtColor(cv2.imread(str(ROOT/f'output/metric_refit/raw_initial/{sim_name}.png')),cv2.COLOR_BGR2RGB)/255
        values=response[sim_name]
        target=expit((logit(np.clip(real,.001,.999))-values['log_exposure_bias'])/values['contrast'])**2.2
        for name,green in [('pla',True),('red',False)]:
            a=object_mask(np.uint8(real*255),green);b=object_mask(np.uint8(sim*255),green)
            if a is None or b is None:continue
            mask=cv2.erode((a[0]&b[0]).astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
            ratio=target[mask].mean(0)/(sim[mask]**2.2).mean(0)
            measurements[name].append(dict(camera=camera,matched_pixels=int(mask.sum()),linear_ratio=ratio.tolist()))
    result=dict(method='fixed linear reflectance estimates from both aligned initial object views',
        source='green_cube_red_plate_1 episode 0 frame 0; per-camera response inverted before fitting',
        colors={},measurements=measurements,
        assumptions=['One material per object for all poses and both sensors',
            'RGB photographic estimate; lighting/ISP ambiguity remains',
            'No reference photo is applied as a texture'])
    for name,records in measurements.items():
        # Balance both cameras rather than fitting a per-camera object color.
        factor=np.exp(np.mean(np.log([r['linear_ratio'] for r in records]),axis=0))
        result['colors'][name]=np.clip(np.array(base_colors[name])*np.clip(factor,.1,2.),.0005,.95).tolist()
    (ROOT/'assets/video_materials.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
