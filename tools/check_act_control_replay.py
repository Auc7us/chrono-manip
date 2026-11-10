#!/usr/bin/env python3
"""Diagnostic: replay recorded next-tick targets with ACT's 30 Hz control timing.

This is an expert-action replay, never a learned-policy evaluation.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    import pychrono.core as chrono
    from demo_attempt_codex import Scene,parse_args
    from scene_config import WORK_SURFACE_Z
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('episode',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--interpolation',choices=['linear','pchip','cubic'],default='linear');a=p.parse_args()
    info=json.loads((a.episode/'episode.json').read_text());data=np.load(a.episode/'trajectory.npz')
    fps=1/np.median(np.diff(data['timestamp']));command=np.rad2deg(data['observation_state'][0])
    cfg=parse_args(['--auto','--headless','--no-sensors','--no-realtime','--tray-style',info['tray_geometry']['style'],
        '--joint-deg',*[str(x) for x in command]])
    scene=Scene(chrono,json.loads((ROOT/'assets/scene_assets.json').read_text()),cfg)
    scene.tray.SetPos(scene.vec([*info['tray_xy_m'],WORK_SURFACE_Z]));scene.tray.SetRot(chrono.QuatFromAngleZ(info['tray_yaw_rad']))
    scene.cube.SetPos(scene.vec([*info['cube_spawn_xy_m'],WORK_SURFACE_Z+.0152]));scene.cube.SetRot(chrono.QuatFromAngleZ(info['cube_spawn_yaw_rad']))
    scene.cube.SetPosDt(scene.vec([0,0,0]));scene.cube.SetAngVelParent(scene.vec([0,0,0]));scene.update_arm(0.,command,reset=True)
    interpolator=None
    if a.interpolation!='linear':
        from scipy.interpolate import PchipInterpolator,CubicSpline
        factory=PchipInterpolator if a.interpolation=='pchip' else CubicSpline
        interpolator=factory(np.arange(len(data['action'])+1)/fps,np.vstack([command,np.rad2deg(data['action'])]),axis=0)
    max_error=0.;max_pen=0.;lifted=False
    for index,action in enumerate(np.rad2deg(data['action'])):
        t=float(scene.system.GetChTime());scene.update_arm(t,command)
        max_error=max(max_error,float(np.linalg.norm(np.array(scene.cube_pose()[:3])-data['cube_pose'][index,:3])))
        start=t;end=(index+1)/fps
        while t<end-1e-9:
            if t>start+1e-9:
                target=command+(action-command)*np.clip((t-start)*fps,0,1) if interpolator is None else interpolator(t)
                scene.update_arm(t,target)
            scene.system.DoStepDynamics(min(.001,end-t));t=float(scene.system.GetChTime())
            contacts=scene.jaw_physics.observe();max_pen=max([max_pen]+[v['penetration_m'] for v in contacts.values()])
            lifted|=scene.cube_pose()[2]>WORK_SURFACE_Z+.06
        command=action
    final=np.array(scene.cube_pose());reference=np.array(info['cube_final_pose'])
    result=dict(method=f'Recorded expert actions with {a.interpolation} 30 Hz next-tick interpolation; no learned policy',
        source_episode=str(a.episode),maximum_cube_position_difference_m=max_error,
        final_cube_position_difference_m=float(np.linalg.norm(final[:3]-reference[:3])),
        final_cube_pose=final.tolist(),reference_final_cube_pose=reference.tolist(),lifted=bool(lifted),maximum_jaw_penetration_m=max_pen)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
