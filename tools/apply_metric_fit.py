#!/usr/bin/env python3
"""Install a reviewed shared fit into the demo's runtime calibration assets."""
import argparse
import json
from pathlib import Path
import numpy as np
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('fit',type=Path)
    a=p.parse_args();report=json.loads(a.fit.read_text());assets=ROOT/'assets'
    from reference_camera import validate_lens
    validate_lens(report['wrist_lens'])
    common=dict(method=report['method'],validation=report['after'],assumptions=report['assumptions'])
    mapping=dict(**common,source='green_cube_red_plate_1; shared multi-episode fit',
        scale=report['scale'],offset_deg=report['offset_deg'],units='cad_degrees = recorded_state * scale + offset_deg')
    if 'encoder_scale_constraint' in report:mapping['encoder_scale_constraint']=report['encoder_scale_constraint']
    for key in ['motor_calibration','home_prior_std_deg','offset_correction_from_nominal_home_deg','floor_clearance_constraint']:
        if key in report:mapping[key]=report[key]
    (assets/'video_joint_map.json').write_text(json.dumps(mapping,indent=2)+'\n')
    wrist=dict(**common,source='green_cube_red_plate_1 wrist chunk; fixed jaw and moving scene features',
        link_from_optical=report['wrist_link_from_optical'],lens=report['wrist_lens'])
    (assets/'video_wrist.json').write_text(json.dumps(wrist,indent=2)+'\n')
    scene=json.loads((assets/'scene_assets.json').read_text())
    base=np.eye(4);base[:3,:3]=np.diag([-1.,-1.,1.]);base[:3,3]=[0.,.054,-.00039]+np.array(report['base_translation_delta_m'])
    scene['base_world_from_cad']=base.tolist();scene['base_calibration']=dict(method=report['method'],
        floor_constraint='CAD feet remain within 0.15 mm of the measured floor',translation_delta_m=report['base_translation_delta_m'])
    (assets/'scene_assets.json').write_text(json.dumps(scene,indent=2)+'\n')
    print('Installed shared motor, base and independent wrist calibration.')


if __name__=='__main__':main()
