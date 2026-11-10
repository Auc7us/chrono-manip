#!/usr/bin/env python3
"""Replay recorded observation.state and save synchronized real/Chrono videos.

Preparation needs pyarrow and OpenCV; rendering uses the Chrono environment.
The cube is triangulated visually, with plane fallback when occluded. This is
kinematic visual replay, not a validation of grasp physics or motor calibration.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def prepare(dataset,output):
    import pyarrow.parquet as pq
    info=json.loads((dataset/'meta/info.json').read_text())
    episodes=[r for f in sorted((dataset/'meta/episodes').rglob('*.parquet'))
              for r in pq.read_table(f).to_pylist()]
    rows=[]
    for file in sorted((dataset/'data').rglob('*.parquet')):
        rows.extend(pq.read_table(file,columns=['observation.state','action','timestamp','episode_index','index']).to_pylist())
    rows.sort(key=lambda r:r['index'])
    arrays=dict(observation_state=np.array([r['observation.state'] for r in rows]),
        action=np.array([r['action'] for r in rows]),
        timestamp=np.array([r['timestamp'] for r in rows]),
        episode_index=np.array([r['episode_index'] for r in rows]),index=np.array([r['index'] for r in rows]))
    metadata=dict(dataset=str(dataset.resolve()),fps=info['fps'],
        joint_names=info['features']['observation.state']['names'],episodes=[],
        data_sha256={str(f.relative_to(dataset)):hashlib.sha256(f.read_bytes()).hexdigest()
                     for f in sorted((dataset/'data').rglob('*.parquet'))})
    for ep in episodes:
        videos={}
        for cam in ['external','wrist']:
            key='observation.images.'+cam;prefix='videos/'+key
            path=info['video_path'].format(video_key=key,chunk_index=ep[prefix+'/chunk_index'],file_index=ep[prefix+'/file_index'])
            videos[cam]=dict(file=path,from_timestamp=ep[prefix+'/from_timestamp'])
        metadata['episodes'].append(dict(episode_index=ep['episode_index'],length=ep['length'],videos=videos))
    output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output,**arrays,metadata=np.array(json.dumps(metadata)))
    print(f'Prepared {len(rows)} observed states from {len(episodes)} episodes: {output}')


def read_video(cap,index):
    import cv2
    if round(cap.get(cv2.CAP_PROP_POS_FRAMES))!=index:cap.set(cv2.CAP_PROP_POS_FRAMES,index)
    ok,frame=cap.read()
    if not ok:raise RuntimeError(f'Cannot decode reference frame {index}')
    return cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)


def color_center(image,green=True):
    import cv2
    f=image.astype(float);r,g,b=f[:,:,0],f[:,:,1],f[:,:,2]
    mask=((g>r*1.5)&(g>b*1.25)&(g>20)) if green else ((r>g*1.7)&(r>b*1.5)&(r>60))
    n,labels,stats,centers=cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    candidates=[i for i in range(1,n) if stats[i,cv2.CC_STAT_AREA]>12]
    if not candidates:return None
    i=max(candidates,key=lambda i:stats[i,cv2.CC_STAT_AREA])
    return centers[i]


def render(a):
    import cv2
    import pychrono.core as chrono
    import pychrono.sensor as sens
    from PIL import Image,ImageDraw
    from demo_attempt_codex import Scene,parse_args,configure_sensors
    from episode_data import capture_synchronized
    from reference_camera import pixel_rays
    data=np.load(a.trajectory);metadata=json.loads(str(data['metadata']))
    ep=next(ep for ep in metadata['episodes'] if ep['episode_index']==a.episode)
    selected=np.flatnonzero(data['episode_index']==a.episode)
    selected=selected[a.start_frame:a.start_frame+a.frames if a.frames else None:a.stride]
    if not len(selected):raise ValueError('No frames selected')
    if a.output.exists():raise FileExistsError(f'Choose a new replay output: {a.output}')
    a.output.mkdir(parents=True)
    config=parse_args(['--headless','--no-realtime','--rate',str(metadata['fps']/a.stride),
        '--camera-profile',str(a.camera_profile),'--samples',str(a.samples)]+(['--raw-response'] if a.raw_response else []))
    assets=json.loads((ROOT/'assets/scene_assets.json').read_text())
    joint_map=json.loads(a.joint_map.read_text())
    scene=Scene(chrono,assets,config,simulate_jaw=False)
    scene.cube.SetFixed(True);scene.cube.EnableCollision(False)
    scene.cube.SetRot(scene.quaternion(np.eye(3)))
    scene.tray.SetRot(scene.quaternion(np.eye(3)))
    manager,sensors=configure_sensors(scene,sens)
    dataset=a.dataset if a.dataset.is_dir() else Path(metadata['dataset'])
    captures={name:cv2.VideoCapture(str(dataset/v['file'])) for name,v in ep['videos'].items()}
    if not all(cap.isOpened() for cap in captures.values()):raise RuntimeError('Cannot open both reference videos')
    fps=metadata['fps']/a.stride
    writers={name:cv2.VideoWriter(str(a.output/(name+'_side_by_side.mp4')),cv2.VideoWriter_fourcc(*'mp4v'),fps,(1280,512))
             for name in captures}
    if not all(w.isOpened() for w in writers.values()):raise RuntimeError('Cannot create comparison videos')
    records=[];errors={k:[] for k in captures};last_cube=None;last_tray=None
    try:
        for count,row in enumerate(selected):
            stamp=float(data['timestamp'][row]);t=count/fps
            q=data['observation_state'][row]*np.array(joint_map['scale'])+joint_map['offset_deg']
            scene.update_arm(t,q,reset=True);scene.system.SetChTime(t)
            real={name:read_video(cap,round((ep['videos'][name]['from_timestamp']+stamp)*metadata['fps']))
                  for name,cap in captures.items()}
            profiles={name:{**scene.camera_profile,'lens':scene.wrist_fit['lens'] if name=='wrist' else scene.camera_profile['lens'],
                'optical_position_m':scene.camera_world_pose('wrist' if name=='wrist' else 'right_45')[:3,3],
                'world_from_optical_rotation':scene.camera_world_pose('wrist' if name=='wrist' else 'right_45')[:3,:3]} for name in captures}
            centers={name:color_center(frame) for name,frame in real.items()}
            cube_mode='held_previous_estimate'
            if centers['external'] is not None:
                p=profiles['external'];o=np.array(p['optical_position_m']);d=pixel_rays([centers['external']],p)[0]
                cube=o+d*((.016-o[2])/d[2]);cube_mode='external_floor_plane'
                if centers['wrist'] is not None:
                    w=profiles['wrist'];ow=w['optical_position_m'];dw=pixel_rays([centers['wrist']],w)[0]
                    along=np.linalg.lstsq(np.column_stack([d,-dw]),ow-o,rcond=None)[0]
                    # Keep the wrist centroid on its measured ray; it resolves
                    # a much larger object than the distant external camera.
                    candidate=ow+dw*along[1]
                    if all(along>0) and -.01<candidate[2]<.3:
                        cube=candidate;cube_mode='two_view_centroid_triangulation'
                cube[2]=max(.016,cube[2]);last_cube=cube
            if last_cube is not None:scene.cube.SetPos(scene.vec(last_cube))
            red=color_center(real['external'],green=False)
            if red is not None:
                p=profiles['external'];o=np.array(p['optical_position_m']);d=pixel_rays([red],p)[0]
                last_tray=o+d*((.004-o[2])/d[2]);last_tray[2]=.001
            if last_tray is not None:scene.tray.SetPos(scene.vec(last_tray))
            images,stamps,launches=capture_synchronized(scene.system,manager,sensors,t)
            frame_record=dict(dataset_index=int(data['index'][row]),episode_timestamp=stamp,
                replay_timestamp=t,cad_joint_deg=q.tolist(),sensor_timestamps=stamps,
                sensor_launches=launches,
                world_from_optical={name:scene.camera_world_pose('wrist' if name=='wrist' else 'right_45').tolist() for name in captures},
                cube_mode=cube_mode,cube_center_m=None if last_cube is None else last_cube.tolist())
            for name in captures:
                sim=images['wrist' if name=='wrist' else 'right_45'];actual=real[name]
                errors[name].append(float(np.abs(sim.astype(float)-actual).mean()))
                canvas=Image.new('RGB',(1280,512),(24,28,33));canvas.paste(Image.fromarray(actual),(0,32));canvas.paste(Image.fromarray(sim),(640,32))
                draw=ImageDraw.Draw(canvas);draw.text((12,9),f'REAL {name} | episode {a.episode} | {stamp:.3f} s',fill='white')
                draw.text((652,9),'VISUAL REPLAY | cube follows image estimates',fill='white')
                writers[name].write(cv2.cvtColor(np.array(canvas),cv2.COLOR_RGB2BGR))
                if count%max(1,round(fps*3))==0:
                    Image.fromarray(sim).save(a.output/f'{name}_sim_{count:05}.png')
                    Image.fromarray(actual).save(a.output/f'{name}_real_{count:05}.png')
                    canvas.save(a.output/f'{name}_comparison_{count:05}.jpg')
            records.append(frame_record)
            if count%30==0:print(f'Replayed {count+1}/{len(selected)} frames',flush=True)
    finally:
        for writer in writers.values():writer.release()
        for cap in captures.values():cap.release()
        sensors.clear();del manager
    if shutil.which('ffmpeg'):
        for name in captures:
            target=a.output/(name+'_side_by_side.mp4');temporary=a.output/(name+'_h264.mp4')
            subprocess.run(['ffmpeg','-y','-loglevel','error','-threads','2','-i',str(target),
                '-c:v','libx264','-threads','2','-preset','fast','-crf','18','-pix_fmt','yuv420p',
                '-movflags','+faststart',str(temporary)],check=True)
            temporary.replace(target)
    from appearance import appearance_metadata
    report=dict(mode='kinematic_visual_replay',source=metadata,episode=a.episode,frames=len(records),fps=fps,
        camera_profile=str(a.camera_profile),optical_profile=scene.camera_profile,joint_map=joint_map,
        appearance=appearance_metadata(config),
        metric='complete-frame RGB8 MAE; no image registration, crop or color matching',
        mean_rgb_mae={k:float(np.mean(v)) for k,v in errors.items()},records=records,
        limitations=['Motor mapping and wrist optical pose are visually estimated',
            'Cube trajectory uses color-centroid triangulation with floor/hold fallbacks; orientation is not tracked',
            'Tray center is visually estimated; tray orientation is not tracked',
            'Kinematic replay does not test physical grasps or calibrated arm dynamics'])
    (a.output/'replay.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k in ['frames','fps','mean_rgb_mae']},indent=2))


def main():
    from reference_camera import DEFAULT_PROFILE
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,default=ROOT.parent/'ref_data/green_cube_red_plate_1')
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--trajectory',type=Path,default=ROOT/'output/video_fit/reference_trajectory.npz')
    p.add_argument('--output',type=Path,default=ROOT/'output/video_fit/replay')
    p.add_argument('--episode',type=int,default=0);p.add_argument('--start-frame',type=int,default=0)
    p.add_argument('--frames',type=int);p.add_argument('--stride',type=int,default=2)
    p.add_argument('--samples',type=int,default=2);p.add_argument('--raw-response',action='store_true')
    p.add_argument('--camera-profile',type=Path,default=DEFAULT_PROFILE)
    p.add_argument('--joint-map',type=Path,default=ROOT/'assets/video_joint_map.json')
    a=p.parse_args()
    if a.stride<1 or a.start_frame<0 or (a.frames is not None and a.frames<1):p.error('Positive stride/frames and nonnegative start-frame required')
    if a.prepare_only:prepare(a.dataset,a.trajectory)
    else:render(a)


if __name__=='__main__':main()
