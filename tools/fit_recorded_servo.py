#!/usr/bin/env python3
"""Fit unloaded gripper response timing from recorded commands and positions."""
import json
import sys
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares

ROOT=Path(__file__).resolve().parents[1]


def main():
    data=np.load(ROOT/'output/video_fit/reference_trajectory.npz')
    mapping=json.loads((ROOT/'assets/video_joint_map.json').read_text());scale=mapping['scale'][5];offset=mapping['offset_deg'][5]
    hardware_path=ROOT/'assets/recorded_motor_calibration.json'
    commands=data['action']
    if hardware_path.exists():
        sys.path.insert(0,str(ROOT))
        from motor_calibration import quantize_commands
        commands=quantize_commands(commands,json.loads(hardware_path.read_text()),recover_leader_grid=True)
    rows=[]
    for ep in range(10):
        sel=np.flatnonzero(data['episode_index']==ep);time=data['timestamp'][sel]
        observed=data['observation_state'][sel,5]*scale+offset;command=commands[sel,5]*scale+offset
        # Open gaps exclude the cube-loaded stall during grasp. Finite-difference
        # velocities correspond to the action issued at the start of that tick.
        mask=(data['observation_state'][sel[:-1],5]>25)&(np.diff(time)>.02)
        rows.append((ep,time,observed,command,mask))
    def residual(p,delay,held_out=False):
        errors=[]
        for ep,t,q,command,mask in rows:
            if (ep>=8)!=held_out:continue
            target=np.interp(t[:-1]-delay,t,command)
            error=target-q[:-1];tau=np.where(error>=0,p[0],p[1])
            predicted=np.clip(error/tau,-p[3],p[2])
            measured=np.diff(q)/np.diff(t)
            errors.extend((predicted-measured)[mask])
        return np.array(errors)
    best=None
    for delay in np.linspace(0,.15,19):
        fit=least_squares(lambda p:residual(p,delay),[.08,.08,180.,180.],
            bounds=([.005,.005,30,30],[.6,.6,600,600]),loss='soft_l1',f_scale=15,max_nfev=100)
        if best is None or fit.cost<best[0]:best=(fit.cost,delay,fit)
    _,delay,fit=best;p=fit.x
    report=dict(method='unloaded command-to-observed velocity response; episodes 0-7 fit, 8-9 validate',
        commands_quantized_to_follower_ticks=hardware_path.exists(),
        close_time_constant_s=float(p[0]),open_time_constant_s=float(p[1]),
        command_latency_s=float(delay),speed_limits_identified=False,
        candidate_speed_limits_used_in_fit_deg_s=p[2:4].tolist(),
        unloaded_threshold_recorded_percent=25.,training_velocity_rms_deg_s=float(np.sqrt(np.mean(residual(p,delay)**2))),
        held_out_velocity_rms_deg_s=float(np.sqrt(np.mean(residual(p,delay,True)**2))),
        previous_held_out_velocity_rms_deg_s=float(np.sqrt(np.mean(residual([.015,.015,458.4,458.4],0,True)**2))),
        assumptions=['Fit uses measured motion in open gaps, excluding the cube-loaded stall',
            'Torque limit, damping, friction and compliance remain unmeasured',
            'The velocity controller retains physical torque saturation and contact-determined actual jaw motion'])
    (ROOT/'output/metric_refit/recorded_servo_candidate.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
