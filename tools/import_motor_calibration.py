#!/usr/bin/env python3
"""Audit the supplied follower/leader calibrations against every dataset row."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from motor_calibration import NAMES,SIGNS,DEG_PER_TICK,to_ticks,nominal_home_offsets


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--calib',type=Path,default=ROOT.parent/'calib')
    p.add_argument('--trajectory',type=Path,default=ROOT/'output/video_fit/reference_trajectory.npz')
    a=p.parse_args();data=np.load(a.trajectory);report=dict(method='supplied LeRobot calibration cross-checked against all recorded encoder values',rows=len(data['observation_state']))
    for role,file,column in [('follower','robots/so_follower/so101_follower.json','observation_state'),
                             ('leader','teleoperators/so_leader/so101_leader.json','action')]:
        path=a.calib/file;raw=path.read_bytes();cal=json.loads(raw);values=data[column]
        tests={}
        for mode in ['degrees','range_m100_100']:
            ticks=to_ticks(values,cal,mode);error=abs(ticks-np.rint(ticks))
            tests[mode]=np.quantile(error,.99,axis=0).tolist()
        if max(tests['degrees'])>.001:raise ValueError(f'{role} calibration does not explain recorded integer encoder readings')
        ticks=to_ticks(values,cal)
        report[role]=dict(source=str(path.relative_to(ROOT.parent)),sha256=hashlib.sha256(raw).hexdigest(),
            calibration=cal,body_normalization='degrees',gripper_normalization='range_0_100',
            grid_99th_percentile_error_ticks=tests,decoded_tick_min=ticks.min(0).tolist(),decoded_tick_max=ticks.max(0).tolist())
    cal=report['follower']['calibration'];span=cal['gripper']['range_max']-cal['gripper']['range_min']
    signed=SIGNS.copy();signed[5]*=span*DEG_PER_TICK/100
    if cal['gripper']['drive_mode']:signed[5]*=-1
    report['signed_scale']=signed.tolist()
    report['nominal_home_offset_deg']=nominal_home_offsets(cal).tolist()
    report['home_tick_convention_uncertainty']=[2047,2048]
    mapping=json.loads((ROOT/'assets/video_joint_map.json').read_text())
    report['visual_offset_minus_nominal_home_deg']=(np.array(mapping['offset_deg'][:5])-report['nominal_home_offset_deg']).tolist()
    quantized=__import__('motor_calibration').quantize_commands(data['action'],report,recover_leader_grid=True)
    error=(quantized-data['action'])*signed
    report['command_quantization_max_abs_deg']=abs(error).max(0).tolist()
    report['notes']=['Homing offsets are already applied in motor firmware; never add them to normalized recorded states',
        'Arm degree zero is the midpoint of each saved follower range, not encoder tick 2048',
        'The leader gripper spans 1209 ticks and the follower spans 1518; commands are normalized targets sent through follower calibration',
        'Float32 saved actions are recovered on the verified integer leader grid before reproducing follower command truncation',
        'A hand-set homing pose is not a surveyed CAD joint zero; nominal home offsets are conditional priors',
        'No friction, stiffness, backlash or loaded servo response is encoded in these files']
    report['sources']=['https://github.com/huggingface/lerobot/blob/v0.4.3/src/lerobot/motors/motors_bus.py',
        'https://github.com/huggingface/lerobot/blob/v0.4.3/src/lerobot/robots/so_follower/so_follower.py']
    target=ROOT/'assets/recorded_motor_calibration.json';target.write_text(json.dumps(report,indent=2)+'\n')
    units=json.loads((ROOT/'assets/encoder_units.json').read_text());units['method']=report['method'];units['signed_scale']=signed.tolist()
    units['hardware_calibration']='recorded_motor_calibration.json';units['assumptions']=[report['notes'][0],report['notes'][4]]
    units['sources']=report['sources']
    for j,motor in enumerate(units['motors']):
        motor['normalization']='degrees' if j<5 else 'range_0_100'
        motor['calibration_span_ticks']=cal[NAMES[j]]['range_max']-cal[NAMES[j]]['range_min']
        motor['absolute_grid_99th_percentile_error_ticks']=report['follower']['grid_99th_percentile_error_ticks']['degrees'][j]
        motor['cad_degrees_per_recorded_unit']=signed[j]
    (ROOT/'assets/encoder_units.json').write_text(json.dumps(units,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['rows','signed_scale','nominal_home_offset_deg','visual_offset_minus_nominal_home_deg','command_quantization_max_abs_deg']},indent=2))


if __name__=='__main__':main()
