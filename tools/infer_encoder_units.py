#!/usr/bin/env python3
"""Recover angular scales from the recorded encoder grid, independently of images.

SO follower STS3215 position readings have 4096 values. LeRobot's degree
conversion uses 360/(resolution-1); its gripper always uses percent of the
calibrated tick range. Absolute motor/CAD offsets still need image fitting.
"""
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def main():
    if (ROOT.parent/'calib/robots/so_follower/so101_follower.json').exists() and (ROOT.parent/'calib/teleoperators/so_leader/so101_leader.json').exists():
        from import_motor_calibration import main as import_hardware
        import_hardware()
        return
    states=np.load(ROOT/'output/video_fit/reference_trajectory.npz')['observation_state']
    names=['shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll','gripper']
    signs=[-1,1,1,1,-1,-1];motors=[]
    for j,name in enumerate(names):
        normalized_range=100 if j==5 else 200
        differences=states[:,j]-states[0,j];candidates=[]
        for ticks in range(500,4096):
            quantum=normalized_range/ticks;grid=differences/quantum
            error=np.abs(grid-np.round(grid))
            if np.quantile(error,.99)<.001:candidates.append((ticks,quantum,float(np.quantile(error,.99))))
        if not candidates:raise ValueError(f'No consistent encoder grid for {name}')
        ticks,quantum,error=candidates[0]
        scale=signs[j]*(360/4095)/quantum
        motors.append(dict(name=name,inferred_calibration_span_ticks=ticks,
            recorded_units_per_encoder_tick=quantum,grid_99th_percentile_error_ticks=error,
            cad_degrees_per_recorded_unit=scale))
    report=dict(method='integer encoder-grid inference from all 6632 observed states',motors=motors,
        signed_scale=[m['cad_degrees_per_recorded_unit'] for m in motors],
        sources=['https://github.com/huggingface/lerobot/blob/main/src/lerobot/motors/motors_bus.py',
                 'https://github.com/huggingface/lerobot/blob/main/src/lerobot/robots/so_follower/so_follower.py'],
        assumptions=['12-bit STS3215 position encoder, matching the supplied SO follower model',
            'First five columns are consistent with degrees or an equivalent 2275-tick normalized span; either interpretation yields the same angular scale',
            'Gripper remains a calibrated percentage; its inferred span is 1518 ticks',
            'Signs follow CAD axes; absolute offsets are fitted separately'])
    (ROOT/'assets/encoder_units.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
