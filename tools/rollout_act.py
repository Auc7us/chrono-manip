#!/usr/bin/env python3
"""Run learned ACT actions through the Chrono contact model; no expert control."""
import argparse
import gc
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from act_inference import load_act
from so101_format import convert_joints


def placement_state(scene):
    relative=np.linalg.inv(scene.body_transform(scene.tray))@scene.body_transform(scene.cube)
    corners=np.array([[x,y,z,1.] for x in [-.015,.015] for y in [-.015,.015] for z in [-.015,.015]])
    local=(relative@corners.T).T[:,:3]
    inside=bool(np.max(abs(local[:,:2]))<.0235 and abs(relative[2,3]-.017)<.002)
    velocity=scene.cube.GetPosDt();speed=math.sqrt(velocity.x**2+velocity.y**2+velocity.z**2)
    return relative,local,inside,speed


def main():
    import cv2
    import pychrono.core as chrono
    import pychrono.sensor as sens
    from demo_attempt_codex import Scene,parse_args,configure_sensors
    from episode_data import capture_synchronized
    from pick_place import Layout,sample_layout
    from so101_ik import START_POSES_DEG,LIMITS_DEG
    from scene_config import WORK_SURFACE_Z
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--episodes',type=int,default=10);p.add_argument('--seed',type=int,default=1000)
    p.add_argument('--training-dataset',type=Path,default=ROOT/'output/act_data/train')
    p.add_argument('--source',type=Path,default=ROOT/'output/pick_and_place')
    p.add_argument('--held-out-split',type=Path,help='Use held-out source layouts instead of fresh seeded layouts')
    p.add_argument('--layouts-from',type=Path,help="Replay the initial layouts of a previous rollout's results.json, e.g. to compare two policies")
    p.add_argument('--horizon',type=float,default=20.);p.add_argument('--settle',type=float,default=1.)
    p.add_argument('--action-steps',type=int,help='Inference-only chunk execution length; weights and normalization stay unchanged')
    ensemble=p.add_mutually_exclusive_group()
    ensemble.add_argument('--temporal-ensemble-coeff',type=float,default=.01,help='Use standard ACT overlapping-chunk ensembling, with a new inference every frame (default .01)')
    ensemble.add_argument('--no-temporal-ensemble',dest='temporal_ensemble_coeff',action='store_const',const=None)
    p.add_argument('--record-episodes',type=int,default=3);p.add_argument('--max-joint-speed',type=float,default=180.)
    p.add_argument('--interpolation',choices=['linear','pchip'],default='linear',
        help='1 ms bridge between 30 Hz targets; pchip fits previous, current, next and the ensemble\'s following target')
    p.add_argument('--release-timeout',type=float,default=1.,help='seconds after the first release for the cube to settle in the tray before the episode fails')
    p.add_argument('--live-frame',type=Path,help='overwrite this .ppm with the latest global (right_45) camera frame, plus a .json status, every frame')
    a=p.parse_args()
    if a.interpolation=='pchip' and a.temporal_ensemble_coeff is None:
        p.error('--interpolation pchip reads the following target from the temporal ensemble')
    from scipy.interpolate import PchipInterpolator
    if a.action_steps not in [None,1] and a.temporal_ensemble_coeff is not None:
        p.error('Use --no-temporal-ensemble to choose a multi-frame --action-steps value')
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    metadata=json.loads((a.training_dataset/'chrono_metadata.json').read_text());conversion=metadata['so101_conversion'];mapping=conversion['cad_joint_map']
    source=json.loads((a.source/'dataset.json').read_text());fps=int(source['fps']);width=source['camera_config']['width'];height=source['camera_config']['height']
    policy,pre,post=load_act(a.checkpoint,action_steps=a.action_steps,temporal_ensemble_coeff=a.temporal_ensemble_coeff);assets=json.loads((ROOT/'assets/scene_assets.json').read_text())
    source_layouts=[]
    if a.held_out_split:
        selected=set(json.loads(a.held_out_split.read_text())['validation_indices']);seen=set();extra=[]
        for e in source['episodes']:
            if e['index'] not in selected:continue
            key=(e.get('worker_id',0),e.get('worker_episode_index',e['index'])//source['tray_every'])
            if key in seen:extra.append(e)
            else:seen.add(key);source_layouts.append(e)
        source_layouts+=extra
        if a.episodes>len(source_layouts):raise ValueError('Not enough held-out layouts')
    if a.layouts_from:
        previous=json.loads(a.layouts_from.read_text());previous=previous['episodes'] if isinstance(previous,dict) else previous
        replayed=[e['initial_layout'] for e in sorted(previous,key=lambda e:e['episode'])]
        if a.episodes>len(replayed):raise ValueError('Not enough layouts in --layouts-from')
    results=[]
    for episode in range(a.episodes):
        seed=a.seed+episode;home=START_POSES_DEG[source.get('start_pose','folded')].copy()
        cfg=parse_args(['--auto','--headless','--no-realtime','--rate',str(fps),'--width',str(width),
            '--global-camera','right-45','--tray-style',source['tray_geometry']['style'],'--joint-deg',*[str(v) for v in home]])
        scene=Scene(chrono,assets,cfg)
        original=None
        if a.held_out_split:
            entry=source_layouts[episode];original=entry['path'];info=json.loads((a.source/original/'episode.json').read_text())
            layout=Layout(np.array(info['cube_spawn_xy_m']),info['cube_spawn_yaw_rad'],np.array(info['tray_xy_m']),info['tray_yaw_rad'])
        elif a.layouts_from:
            l=replayed[episode];layout=Layout(np.array(l['cube_xy_m']),l['cube_yaw_rad'],np.array(l['tray_xy_m']),l['tray_yaw_rad'])
        else:
            # IK is used only to sample a reachable initial layout. It never
            # produces a control action during the learned-policy rollout.
            layout=sample_layout(assets,np.random.default_rng(seed),tray_style=cfg.tray_style,planner=source.get('planner','direct')).layout
        scene.tray.SetPos(scene.vec([*layout.tray_xy,WORK_SURFACE_Z]));scene.tray.SetRot(chrono.QuatFromAngleZ(layout.tray_yaw))
        scene.cube.SetPos(scene.vec([*layout.cube_xy,WORK_SURFACE_Z+.0152]));scene.cube.SetRot(chrono.QuatFromAngleZ(layout.cube_yaw))
        scene.cube.SetPosDt(scene.vec([0,0,0]));scene.cube.SetAngVelParent(scene.vec([0,0,0]));scene.update_arm(0.,home,reset=True)
        manager,sensors=configure_sensors(scene,sens);policy.reset();command=home.copy();previous=home.copy()
        folder=a.output/f'episode_{episode:03d}';folder.mkdir();video=None
        if episode<a.record_episodes:
            video=cv2.VideoWriter(str(folder/'cameras_raw.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,(width*2,height+32))
            if not video.isOpened():raise RuntimeError('Video writer did not open')
        rows=[];grasped=False;lifted=False;max_force=0.;max_penetration=0.;max_environment=0.;max_height=scene.cube_pose()[2];clipped=0;started=time.monotonic()
        frames=round((a.horizon+a.settle)*fps);termination='horizon_reached';stable_placement_frames=0;released_at=None
        try:
            for frame in range(frames):
                t=float(scene.system.GetChTime());scene.update_arm(t,command)
                images,stamps,launches=capture_synchronized(scene.system,manager,sensors,t)
                if a.live_frame:
                    # Atomic replace, so a viewer never reads a half-written frame.
                    partial=a.live_frame.with_suffix('.partial.ppm');cv2.imwrite(str(partial),cv2.cvtColor(images['right_45'],cv2.COLOR_RGB2BGR))
                    partial.replace(a.live_frame)
                    a.live_frame.with_suffix('.json').write_text(json.dumps(dict(episode=episode,episodes=a.episodes,seed=seed,t=round(t,2),
                        done=len(results),clean=sum(r['success'] for r in results),placed=sum(r['placement_success'] for r in results))))
                state=convert_joints(np.deg2rad(scene.joint_deg),conversion).astype(np.float32)
                raw=None
                if frame/fps<a.horizon:
                    observation={'observation.state':torch.from_numpy(state)}
                    for camera,alias in conversion['camera_names'].items():
                        observation['observation.images.'+alias]=torch.from_numpy(images[camera].copy()).permute(2,0,1).float()/255.
                    with torch.inference_mode():raw=post(policy.select_action(pre(observation))).cpu().numpy().reshape(6)
                    if not np.isfinite(raw).all():raise ValueError('Policy produced a nonfinite action')
                    target=raw*np.array(mapping['scale'])+np.array(mapping['offset_deg'])
                    bounded=np.clip(target,LIMITS_DEG[:,0],LIMITS_DEG[:,1])
                    if a.max_joint_speed>0:bounded=command+np.clip(bounded-command,-a.max_joint_speed/fps,a.max_joint_speed/fps)
                    clipped+=int(not np.allclose(target,bounded,atol=1e-6,rtol=0));target=bounded
                    following=target
                    ahead=policy.temporal_ensembler.ensembled_actions if a.interpolation=='pchip' else None
                    if ahead is not None and ahead.shape[1]:
                        with torch.inference_mode():following=post(ahead[:,0]).cpu().numpy().reshape(6)
                        following=np.clip(following*np.array(mapping['scale'])+np.array(mapping['offset_deg']),LIMITS_DEG[:,0],LIMITS_DEG[:,1])
                else:target=command.copy();following=target
                rows.append(dict(timestamp=t,observation_state_so101=state.tolist(),raw_action_so101=None if raw is None else raw.tolist(),
                    executed_next_target_cad_deg=target.tolist(),cube_pose=scene.cube_pose(),sensor_timestamps=stamps,sensor_launches=launches))
                if video:
                    canvas=np.zeros((height+32,width*2,3),np.uint8)
                    for column,name in enumerate(['wrist','right_45']):canvas[32:,column*width:(column+1)*width]=images[name]
                    cv2.putText(canvas,f'ACT | episode {episode} | {t:.2f} s | wrist / external',(12,22),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1)
                    video.write(cv2.cvtColor(canvas,cv2.COLOR_RGB2BGR))
                    if frame==0:cv2.imwrite(str(folder/'initial_cameras.png'),cv2.cvtColor(canvas,cv2.COLOR_RGB2BGR))
                    if frame%(fps*2)==0:cv2.imwrite(str(folder/f'frame_{frame:04d}.jpg'),cv2.cvtColor(canvas,cv2.COLOR_RGB2BGR))
                start=t;end=(frame+1)/fps;frame_forces=np.zeros(2);frame_penetration=0.
                spline=PchipInterpolator(np.arange(-1,3),np.vstack([previous,command,target,following]),axis=0) if a.interpolation=='pchip' else None
                while t<end-1e-9:
                    # The frame boundary was already applied before capture.
                    if t>start+1e-9:
                        fraction=np.clip((t-start)*fps,0,1)
                        scene.update_arm(t,command+(target-command)*fraction if spline is None else spline(fraction))
                    scene.system.DoStepDynamics(min(.001,end-t));t=float(scene.system.GetChTime())
                    contacts=scene.jaw_physics.observe();forces=[contacts.get(name,{}).get('normal_force_n',0.) for name in ['fixed_jaw','moving_jaw']]
                    grasped|=min(forces)>.02
                    max_force=max(max_force,*forces);max_penetration=max(max_penetration,*[c['penetration_m'] for c in contacts.values()],0.)
                    frame_forces=np.maximum(frame_forces,forces);frame_penetration=max(frame_penetration,*[c['penetration_m'] for c in contacts.values()],0.)
                    max_environment=max(max_environment,scene.jaw_physics.reporter.environment_penetration)
                    z=scene.cube_pose()[2];max_height=max(max_height,z);lifted|=grasped and z>WORK_SURFACE_Z+.06
                rows[-1].update(max_jaw_normal_force_n=dict(fixed_jaw=frame_forces[0],moving_jaw=frame_forces[1]),max_jaw_penetration_m=frame_penetration)
                previous=command;command=target
                if frame%(fps*2)==0:print(f'Episode {episode}: {t:.1f}/{a.horizon+a.settle:.1f}s, cube z={scene.cube_pose()[2]:.3f}',flush=True)
                position=np.array(scene.cube_pose()[:3])
                if position[2]<WORK_SURFACE_Z-.05 or np.linalg.norm(position[:2])>.75:
                    termination='cube_left_workspace';break
                if grasped and lifted and max(forces)<.01:
                    if released_at is None:released_at=t
                    _,_,placed_now,speed_now=placement_state(scene)
                    stable_placement_frames=stable_placement_frames+1 if placed_now and speed_now<.02 else 0
                else:stable_placement_frames=0
                if stable_placement_frames>=round(.5*fps):
                    termination='released_cube_settled_in_tray';break
                # A release that has not settled in the tray within the timeout is a failure; don't wait for the horizon.
                if released_at is not None and t-released_at>=a.release_timeout:
                    termination='not_settled_in_tray_after_release';break
        finally:
            if video:video.release()
            sensors.clear();del manager;gc.collect()
        relative,local,inside,speed=placement_state(scene)
        released=bool(lifted and max(forces)<.01)
        placed=bool(grasped and lifted and released and inside and speed<.02);quality=bool(max_penetration<.0005 and max_environment<.0005)
        result=dict(episode=episode,seed=seed,source_layout=original,success=placed and quality,placement_success=placed,contact_quality_ok=quality,
            grasped=bool(grasped),lifted=bool(lifted),released=released,cube_in_tray=inside,final_cube_speed_m_s=speed,final_cube_pose=scene.cube_pose(),
            initial_layout=dict(cube_xy_m=layout.cube_xy.tolist(),cube_yaw_rad=float(layout.cube_yaw),tray_xy_m=layout.tray_xy.tolist(),tray_yaw_rad=float(layout.tray_yaw)),
            final_tray_pose=scene.tray_pose(),final_cube_center_in_tray_m=relative[:3,3].tolist(),
            final_cube_footprint_half_extent_m=float(np.max(abs(local[:,:2]))),
            maximum_force_n=max_force,maximum_jaw_penetration_m=max_penetration,maximum_environment_penetration_m=max_environment,
            maximum_cube_height_m=max_height,clipped_control_frames=clipped,frames=len(rows),fps=fps,wall_time_s=time.monotonic()-started,
            termination=termination,maximum_control_horizon_s=a.horizon,settle_s=a.settle,
            expert_actions_used_for_control=False,cube_pose_updates_after_initialization=0,records=rows)
        result['action_steps']=policy.config.n_action_steps
        result['temporal_ensemble_coeff']=policy.config.temporal_ensemble_coeff;result['interpolation']=a.interpolation
        (folder/'rollout.json').write_text(json.dumps(result,indent=2)+'\n')
        results.append({k:v for k,v in result.items() if k!='records'});(a.output/'results.json').write_text(json.dumps(results,indent=2)+'\n')
        if video and shutil.which('ffmpeg'):
            subprocess.run(['ffmpeg','-y','-v','error','-i',str(folder/'cameras_raw.mp4'),'-c:v','libx264','-threads','2','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(folder/'cameras.mp4')],check=True)
        print(json.dumps(results[-1]),flush=True)
        del scene;gc.collect()
    print(f'ACT strict successes: {sum(r["success"] for r in results)}/{len(results)}',flush=True)


if __name__=='__main__':main()
