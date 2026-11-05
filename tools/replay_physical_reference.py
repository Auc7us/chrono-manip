#!/usr/bin/env python3
"""Replay real observed arm motion and commanded jaw targets through contacts.

The cube is initialized once, remains dynamic/colliding, and receives only
gravity and contact forces. Later video object tracks are measurements, never
simulation inputs. The first five arm joints remain prescribed, as in --auto.
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def profiles_for_scene(scene):
    result={}
    for name,sensor in [('external','right_45'),('wrist','wrist')]:
        pose=scene.camera_world_pose(sensor)
        result[name]={**scene.camera_profile,'lens':scene.wrist_fit['lens'] if name=='wrist' else scene.camera_profile['lens'],
            'optical_position_m':pose[:3,3],'world_from_optical_rotation':pose[:3,:3]}
    return result


def main():
    import cv2
    from PIL import Image,ImageDraw
    import pychrono.core as chrono
    from demo_attempt_codex import Scene,parse_args
    from episode_data import capture_synchronized
    from replay_reference import read_video
    from reference_objects import fit_initial_objects,object_mask
    from reference_camera import project_world
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trajectory',type=Path,default=ROOT/'output/video_fit/reference_trajectory.npz')
    parser.add_argument('--dataset',type=Path,default=ROOT.parent/'ref_data/green_cube_red_plate_1')
    parser.add_argument('--episode',type=int,default=0);parser.add_argument('--stride',type=int,default=2)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--no-sensors',action='store_true')
    parser.add_argument('--duration',type=float);parser.add_argument('--samples',type=int,default=2)
    parser.add_argument('--step',type=float,default=.001)
    parser.add_argument('--arm-interpolation',choices=['linear','pchip'],default='linear')
    parser.add_argument('--metric-fit',type=Path,help='Evaluate a candidate calibration without changing runtime assets')
    parser.add_argument('--contact-compliance',type=float,help='Override effective contact compliance, m/N; zero selects rigid contact')
    parser.add_argument('--contact-damping',type=float,default=.002,help='Compliance damping time, seconds')
    parser.add_argument('--jaw-response',type=Path,help='Override the recorded unloaded response profile')
    a=parser.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    data=np.load(a.trajectory);metadata=json.loads(str(data['metadata']))
    if 'action' not in data:raise ValueError('Prepare the trajectory again to include recorded gripper actions')
    ep=next(e for e in metadata['episodes'] if e['episode_index']==a.episode)
    indices=np.flatnonzero(data['episode_index']==a.episode);stamps=data['timestamp'][indices]
    if a.duration:indices=indices[stamps<=a.duration];stamps=data['timestamp'][indices]
    mapping=json.loads((ROOT/'assets/video_joint_map.json').read_text());scale=np.array(mapping['scale']);offset=np.array(mapping['offset_deg'])
    metric_fit=json.loads(a.metric_fit.read_text()) if a.metric_fit else None
    if metric_fit is not None:
        mapping=dict(source_fit=str(a.metric_fit),scale=metric_fit['scale'],offset_deg=metric_fit['offset_deg'],
            validation=metric_fit['after'],assumptions=metric_fit['assumptions'])
        scale=np.array(mapping['scale']);offset=np.array(mapping['offset_deg'])
    observed=data['observation_state'][indices]*scale+offset
    hardware_path=ROOT/'assets/recorded_motor_calibration.json'
    hardware=json.loads(hardware_path.read_text()) if hardware_path.exists() else None
    command_values=data['action'][indices]
    if hardware is not None:
        from motor_calibration import quantize_commands
        np.testing.assert_allclose(scale,hardware['signed_scale'],atol=1e-9,rtol=0)
        command_values=quantize_commands(command_values,hardware,recover_leader_grid=True)
    commands=command_values*scale+offset
    interpolation=None
    if a.arm_interpolation=='pchip':
        from scipy.interpolate import PchipInterpolator
        interpolation=PchipInterpolator(stamps,observed[:,:5],axis=0)
    fps=metadata['fps']/a.stride
    config=parse_args(['--headless','--auto','--no-realtime','--rate',str(fps),'--samples',str(a.samples),
        '--joint-deg',*[str(v) for v in observed[0]]])
    config.contact_compliance_m_n=a.contact_compliance;config.contact_damping_s=a.contact_damping
    if a.jaw_response:config.jaw_response_override=json.loads(a.jaw_response.read_text())
    assets=json.loads((ROOT/'assets/scene_assets.json').read_text())
    if metric_fit is not None:
        base=np.eye(4);base[:3,:3]=np.diag([-1.,-1.,1.]);base[:3,3]=[0.,.054,-.00039]+np.array(metric_fit['base_translation_delta_m']);assets['base_world_from_cad']=base.tolist()
        config.wrist_profile_override=dict(link_from_optical=metric_fit['wrist_link_from_optical'],lens=metric_fit['wrist_lens'])
    scene=Scene(chrono,assets,config)
    scene.update_arm(0.,observed[0],reset=True)
    captures={name:cv2.VideoCapture(str(a.dataset/v['file'])) for name,v in ep['videos'].items()}
    initial_images={name:read_video(cap,round(ep['videos'][name]['from_timestamp']*metadata['fps'])) for name,cap in captures.items()}
    initial=fit_initial_objects(initial_images,profiles_for_scene(scene))
    (a.output/'initial_objects.json').write_text(json.dumps(initial,indent=2)+'\n')
    for name,body in [('cube',scene.cube),('plate',scene.tray)]:
        pose=initial[name];position=pose['center_m'].copy()
        if name=='plate':position[2]=.001
        body.SetPos(scene.vec(position));body.SetRot(scene.quaternion(__import__('demo_attempt_codex').axis_rotation([0,0,1],pose['yaw_rad'])))
        body.SetPosDt(scene.vec([0,0,0]));body.SetAngVelParent(scene.vec([0,0,0]))
    assert not scene.cube.IsFixed() and scene.cube.IsCollisionEnabled()
    manager=None;sensors={};writers={}
    if not a.no_sensors:
        import pychrono.sensor as sens
        from demo_attempt_codex import configure_sensors
        manager,sensors=configure_sensors(scene,sens)
        writers={name:cv2.VideoWriter(str(a.output/(name+'_physical.mp4')),cv2.VideoWriter_fourcc(*'mp4v'),fps,(1280,512)) for name in captures}
    dt=a.step;t=0.;frame=0;records=[];diagnostics=[];max_force=0.;max_penetration=0.;peak_height=initial['cube']['center_m'][2]
    final_time=float(stamps[-1]);next_render=0.;first_contact=None;bilateral_frames=0;image_errors={k:[] for k in captures}
    try:
        while t<=final_time+1e-7:
            q=np.array([np.interp(t,stamps,observed[:,j]) for j in range(6)])
            if interpolation is not None:q[:5]=interpolation(t)
            # Hardware actions can close farther than measured positions while
            # gripping. Let physical contacts determine the actual jaw angle.
            q[5]=np.interp(t,stamps,commands[:,5])
            scene.update_arm(t,q)
            if t>=next_render-1e-7:
                source_index=indices[min(frame*a.stride,len(indices)-1)]
                source_time=float(data['timestamp'][source_index])
                real={name:read_video(cap,round((ep['videos'][name]['from_timestamp']+source_time)*metadata['fps'])) for name,cap in captures.items()}
                contact=scene.jaw_physics.observe();forces=[contact.get(k,{}).get('normal_force_n',0.) for k in ['fixed_jaw','moving_jaw']]
                if all(f>.01 for f in forces):bilateral_frames+=1
                row=dict(dataset_index=int(data['index'][source_index]),timestamp=t,source_timestamp=source_time,
                    observed_arm_joint_deg=q[:5].tolist(),
                    cube_pose=scene.cube_pose(),jaw_actual_deg=scene.joint_deg[5],jaw_target_deg=q[5],
                    jaw_force_n=forces,tcp_pose=np.asarray(scene.tool_world_pose()).tolist(),jaw_torque_nm=scene.jaw_physics.torque)
                row['jaw_filtered_target_deg']=scene.jaw_physics.filtered_target_deg
                profiles=profiles_for_scene(scene)
                for name,image in real.items():
                    item=object_mask(image,True)
                    if item is not None:
                        pixel=project_world([row['cube_pose'][:3]],profiles[name])[0]
                        row[name+'_cube_centroid_error_px']=float(np.linalg.norm(pixel-item[1]))
                if manager:
                    images,sensor_stamps,launches=capture_synchronized(scene.system,manager,sensors,t)
                    row.update(sensor_timestamps=sensor_stamps,sensor_launches=launches)
                    for name in captures:
                        sim=images['wrist' if name=='wrist' else 'right_45'];image_errors[name].append(float(abs(sim.astype(float)-real[name]).mean()))
                        canvas=Image.new('RGB',(1280,512),(24,28,33));canvas.paste(Image.fromarray(real[name]),(0,32));canvas.paste(Image.fromarray(sim),(640,32))
                        draw=ImageDraw.Draw(canvas);draw.text((12,9),f'REAL {name} | episode {a.episode} | {source_time:.3f} s',fill='white')
                        draw.text((652,9),'PHYSICAL REPLAY | dynamic cube, contact forces',fill='white')
                        writers[name].write(cv2.cvtColor(np.array(canvas),cv2.COLOR_RGB2BGR))
                        if frame%max(1,round(fps*3))==0:canvas.save(a.output/f'{name}_{frame:05d}.jpg')
                records.append(row);frame+=1;next_render=frame/fps
                if frame%60==1:print(f'Physical replay {t:.2f}/{final_time:.2f} s, cube z={row["cube_pose"][2]:.4f}',flush=True)
            if t>=final_time-1e-8:break
            step=min(dt,final_time-t,max(1e-8,next_render-t));scene.system.DoStepDynamics(step);t=scene.system.GetChTime()
            contacts=scene.jaw_physics.observe()
            force=max([0.,*[c.get('normal_force_n',0.) for c in contacts.values()]])
            penetration=max([0.,*[c.get('penetration_m',0.) for c in contacts.values()]])
            if force>.01 and first_contact is None:first_contact=t
            max_force=max(max_force,force);max_penetration=max(max_penetration,penetration)
            peak_height=max(peak_height,scene.cube_pose()[2])
            if round(t*1000)%10==0:diagnostics.append([t,*scene.cube_pose(),force,penetration,scene.jaw_physics.torque])
    finally:
        for w in writers.values():w.release()
        for cap in captures.values():cap.release()
        sensors.clear()
        if manager:del manager
    for name in writers:
        source=a.output/(name+'_physical.mp4');temp=a.output/(name+'_h264.mp4')
        if shutil.which('ffmpeg'):
            subprocess.run(['ffmpeg','-y','-loglevel','error','-i',str(source),'-c:v','libx264','-threads','2','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(temp)],check=True);temp.replace(source)
    final=np.array(scene.cube_pose()[:3]);tray=np.array(initial['plate']['center_m']);local=(final[:2]-tray[:2])
    yaw=initial['plate']['yaw_rad'];c,s=np.cos(yaw),np.sin(yaw);local=local@np.array([[c,-s],[s,c]])
    inside=bool((abs(local)<.025-.015).all() and abs(final[2]-.018)<.004)
    # Endpoint vision is a held-out measurement only. It never changes the cube.
    reference_endpoint=fit_initial_objects(real,profiles_for_scene(scene),cube_center_z=.018)['cube']
    endpoint_error=final-np.array(reference_endpoint['center_m'])
    report=dict(mode='physical_recorded_arm_replay',episode=a.episode,frames=len(records),fps=fps,
        physics_step_s=a.step,arm_interpolation=a.arm_interpolation,
        optical_profile=scene.camera_profile,wrist_profile=scene.wrist_fit,joint_map=mapping,
        base_world_from_cad=assets.get('base_world_from_cad'),
        jaw_response_profile=scene.jaw_physics.response,
        hardware_calibration=hardware,commands_quantized_to_follower_ticks=hardware is not None,
        contact_compliance_m_n=scene.jaw_physics.contact_compliance_m_n,contact_damping_s=scene.jaw_physics.contact_damping_s,
        initial_objects=initial,initial_cube_pose_set_count=1,cube_dynamic=True,cube_collision_enabled=True,
        per_frame_cube_pose_updates=False,first_contact_s=first_contact,maximum_contact_force_n=max_force,
        maximum_penetration_m=max_penetration,maximum_cube_height_m=peak_height,bilateral_contact_render_frames=bilateral_frames,
        final_cube_pose=scene.cube_pose(),final_cube_center_offset_from_plate_m=(final[:2]-tray[:2]).tolist(),cube_inside_plate=inside,
        reference_final_cube=reference_endpoint,final_position_error_m=endpoint_error.tolist(),
        final_xy_error_m=float(np.linalg.norm(endpoint_error[:2])),
        mean_rgb_mae={k:float(np.mean(v)) if v else None for k,v in image_errors.items()},records=records,
        limitations=['First five arm joints follow recorded observed positions; jaw driven by recorded commands',
            'Initial object poses fitted once to both real silhouettes; no later vision corrections to physics',
            'Friction, servo torque/gains and printed-part compliance are not measured'])
    (a.output/'physics_replay.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(a.output/'contact_diagnostics.npz',samples=np.array(diagnostics))
    summary_keys=['episode','frames','physics_step_s','arm_interpolation','first_contact_s',
        'maximum_contact_force_n','maximum_penetration_m','maximum_cube_height_m',
        'cube_inside_plate','final_xy_error_m','mean_rgb_mae']
    print(json.dumps({k:report[k] for k in summary_keys},indent=2))


if __name__=='__main__':main()
