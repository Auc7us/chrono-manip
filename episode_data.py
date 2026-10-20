"""Lossless synchronized episode recording, independent of LeRobot installation."""
from pathlib import Path
import json
import os
import shutil
import subprocess
import time
import numpy as np
from so101_ik import JOINT_NAMES
from jaw_physics import GRASP_MODE,JAW_TORQUE_LIMIT,CONTACT_FRICTION,JAW_MASS,JAW_KP,JAW_KD
from scene_config import SHEET_SIZE,SHEET_CENTER,WORK_SURFACE_Z

def capture_synchronized(system,manager,sensors,t):
    """Respect Chrono's float32 launch deadlines without advancing the scene."""
    # ChOptixEngine divides launch count by a float update rate. At 8.1 s,
    # for example, its deadline is 8.100000381 rather than the Python 8.1.
    # Reproduce that deadline instead of relying on a fixed scheduling epsilon.
    deadlines=[float(np.float32(camera.GetNumLaunches())/np.float32(camera.GetUpdateRate()))
               for camera in sensors.values()]
    scheduled=max([t,*deadlines])+1e-6
    try:
        system.SetChTime(scheduled)
        manager.Update()
        return synchronized_images(sensors,float(np.float32(scheduled)))
    finally:
        system.SetChTime(t)

def synchronized_images(sensors,expected_time,timeout=20.):
    """Wait without advancing physics; reject old or mismatched camera frames."""
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        images={};stamps={};launches={}
        for name,camera in sensors.items():
            data=camera.GetMostRecentRGBA8Buffer()
            if not data.HasData() or abs(float(data.TimeStamp)-expected_time)>1e-4:break
            images[name]=np.ascontiguousarray(np.flipud(np.array(data.GetRGBA8Data(),copy=True))[:,:,:3])
            stamps[name]=float(data.TimeStamp);launches[name]=int(data.LaunchedCount)
        if len(images)==len(sensors) and images:
            return images,stamps,launches
        time.sleep(.002)
    observed={name:float(camera.GetMostRecentRGBA8Buffer().TimeStamp) for name,camera in sensors.items()}
    raise RuntimeError(f'Timed out waiting for both cameras at simulation time {expected_time:.6f}; '
                       f'last timestamps {observed}; refusing stale images')

class EpisodeWriter:
    def __init__(self,root,scene,args):
        from PIL import Image
        from appearance import appearance_metadata
        from reference_camera import camera_metadata
        self.Image=Image;self.root=Path(root);self.scene=scene;self.args=args
        if self.root.exists():raise FileExistsError(f'Collection directory already exists: {self.root}; choose a new directory')
        self.root.mkdir(parents=True)
        self.manifest=dict(format='attempt-codex-episodes-v1',fps=args.rate,joint_names=list(JOINT_NAMES),
            joint_units='radians_offset_from_STEP_pose',action_semantics='absolute_joint_target_for_next_control_tick',
            cameras=list(scene.camera_names),grasp_mode=GRASP_MODE,seed=args.seed,
            jaw_collision=scene.jaw_physics.geometry,jaw_torque_limit_nm=JAW_TORQUE_LIMIT,jaw_friction=CONTACT_FRICTION,
            jaw_mass_kg=JAW_MASS,jaw_servo_kp=JAW_KP,jaw_servo_kd=JAW_KD,physics_step_s=.001,
            appearance=appearance_metadata(args),
            work_surface=dict(color='black',dimensions_m=list(SHEET_SIZE),center_m=list(SHEET_CENTER),top_z_m=WORK_SURFACE_Z),
            camera_config=dict(global_view=args.global_camera,width=args.width,height=args.width*3//4,
                hfov_deg=camera_metadata(args)['effective_hfov_deg'],lens=camera_metadata(args),global_world_from_optical=scene.global_optical.tolist(),
                wrist_link_from_optical=scene.wrist_optical.tolist()),
            tray_every=args.tray_every,start_pose=args.start_pose,episodes=[])
        self.active=False
        self.flush()

    def flush(self):
        (self.root/'dataset.json').write_text(json.dumps(self.manifest,indent=2)+'\n')

    def begin(self,controller):
        self.index=controller.episode_index;self.start_time=controller.start_time
        self.directory=self.root/'episodes'/f'{self.index:06d}'
        self.directory.mkdir(parents=True)
        self.rows=[];self.active=True;self.last_launches={}

    def add(self,t,controller,images,stamps,launches):
        scene=self.scene;n=len(self.rows)
        relative=t-self.start_time
        if abs(relative-n/self.args.rate)>1e-5:raise RuntimeError('Nonuniform control timestamps in episode')
        if any(launches[k]<=self.last_launches.get(k,-1) for k in launches):raise RuntimeError('Repeated camera frame in collection')
        self.last_launches=launches.copy()
        q=np.deg2rad(scene.joint_deg)
        velocity=np.zeros(6) if n==0 else (q-self.rows[-1]['state'])*self.args.rate
        for camera,array in images.items():
            target=self.directory/'images'/camera/f'{n:06d}.png';target.parent.mkdir(parents=True,exist_ok=True)
            self.Image.fromarray(array).save(target)
        self.rows.append(dict(timestamp=n/self.args.rate,state=q.copy(),action=controller.next_action(t),
            velocity=velocity,tcp_pose=scene.tool_world_pose(),cube_pose=scene.cube_pose(),tray_pose=scene.tray_pose(),
            image_timestamps=[stamps[k] for k in self.manifest['cameras']],
            image_launches=[launches[k] for k in self.manifest['cameras']],
            grasp_assisted=False,grasp_contact=controller.holding,
            jaw_torque_nm=scene.jaw_physics.torque,
            jaw_contact_force_n=[scene.jaw_physics.contacts.get(k,{}).get('normal_force_n',0.) for k in ['fixed_jaw','moving_jaw']],
            jaw_penetration_m=[scene.jaw_physics.contacts.get(k,{}).get('penetration_m',0.) for k in ['fixed_jaw','moving_jaw']],
            phase=controller.phase))

    def end(self,result):
        if not self.active:return
        rows=self.rows
        if rows:
            np.savez_compressed(self.directory/'trajectory.npz',
                **{key:np.array([r[source] for r in rows]) for key,source in [
                    ('timestamp','timestamp'),('observation_state','state'),('action','action'),('joint_velocity','velocity'),
                    ('tcp_pose','tcp_pose'),('cube_pose','cube_pose'),('tray_pose','tray_pose'),
                    ('image_timestamps','image_timestamps'),('image_launches','image_launches'),
                    ('grasp_assisted','grasp_assisted'),('grasp_contact','grasp_contact'),
                    ('jaw_torque_nm','jaw_torque_nm'),('jaw_contact_force_n','jaw_contact_force_n'),
                    ('jaw_penetration_m','jaw_penetration_m'),('phase','phase')]})
        info={**result,'frames':len(rows),'fps':self.args.rate,'videos':{},
              'task':'Pick up the green cube and place it in the red tray.'}
        # Lossless PNGs remain the canonical input for LeRobot conversion.
        ffmpeg=shutil.which('ffmpeg')
        if ffmpeg and rows:
            for camera in self.manifest['cameras']:
                video=self.directory/f'{camera}.mp4'
                command=[ffmpeg,'-nostdin','-v','error','-framerate',str(self.args.rate),'-i',
                         str(self.directory/'images'/camera/'%06d.png'),'-c:v','libx264','-crf','18','-pix_fmt','yuv420p',str(video)]
                encoder_threads=os.environ.get('ATTEMPT_CODEX_ENCODER_THREADS')
                if encoder_threads:command[-1:-1]=['-threads',str(max(1,int(encoder_threads)))]
                process=subprocess.run(command,capture_output=True,text=True)
                if process.returncode==0:info['videos'][camera]=video.name
                else:info.setdefault('video_encoding_errors',{})[camera]=process.stderr[-1000:]
        (self.directory/'episode.json').write_text(json.dumps(info,indent=2)+'\n')
        self.manifest['episodes'].append(dict(index=self.index,path=str(self.directory.relative_to(self.root)),
            success=bool(result['success']),complete=bool(result['complete']),frames=len(rows)))
        self.flush();self.active=False
