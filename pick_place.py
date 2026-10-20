"""Seeded episode planning with physical jaw contact and grasp validation."""
from dataclasses import dataclass
import json
import math
import numpy as np
from so101_ik import SO101IK, START_POSES_DEG, OPEN_DEG, GRASP_DEG, tool_pose
from jaw_physics import GRASP_MODE
from scene_config import WORK_SURFACE_Z

@dataclass
class Layout:
    cube_xy: np.ndarray
    cube_yaw: float
    tray_xy: np.ndarray
    tray_yaw: float

@dataclass
class Phase:
    name: str
    start: float
    duration: float
    q0: np.ndarray
    q1: np.ndarray
    path: object=None

class EpisodePlan:
    def __init__(self,assets,layout,start_pose='folded'):
        self.layout=layout;self.phases=[];self.ik_results=[]
        self.start_pose=start_pose;self.home_deg=START_POSES_DEG[start_pose].copy()
        ik=SO101IK(assets);q=np.deg2rad(self.home_deg);clock=0.
        def append(name,duration,target,path=None):
            nonlocal q,clock
            self.phases.append(Phase(name,clock,duration,q.copy(),target.copy(),path))
            clock+=duration;q=target.copy()
        def cart(name,duration,xy,z,yaw,jaw):
            start=tool_pose(assets,q);target=np.array([*xy,WORK_SURFACE_Z+z]);path=[q.copy()]
            initial_yaw=math.atan2(start[1,1],start[0,1])
            yaw_delta=(yaw-initial_yaw+math.pi)%(2*math.pi)-math.pi
            fractions=[1.] if name=='approach_cube' else np.linspace(.05,1.,20)
            for fraction in fractions:
                position=start[:3,3]+fraction*(target-start[:3,3])
                result=ik.solve(position,path[-1],yaw=initial_yaw+fraction*yaw_delta,jaw_deg=jaw)
                if not result.success:
                    raise ValueError(f'{name} unreachable: {result.position_error_m*1000:.2f} mm, {result.approach_error_deg:.2f} deg')
                self.ik_results.append(result);path.append(result.q)
            append(name,duration,result.q,np.array(path) if len(path)>2 else None)
        append('settle_start',.4,q)
        cart('approach_cube',1.2,layout.cube_xy,.105,layout.cube_yaw,OPEN_DEG)
        cart('descend_to_cube',1.,layout.cube_xy,.019,layout.cube_yaw,OPEN_DEG)
        closed=q.copy();closed[5]=math.radians(GRASP_DEG);append('close_gripper',.6,closed)
        cart('lift_cube',1.,layout.cube_xy,.105,layout.cube_yaw,GRASP_DEG)
        cart('transfer_to_tray',1.4,layout.tray_xy,.105,layout.tray_yaw,GRASP_DEG)
        # Release above the rim: the jaws' outside width exceeds the 50 mm tray.
        cart('lower_into_tray',1.,layout.tray_xy,.034,layout.tray_yaw,GRASP_DEG)
        opened=q.copy();opened[5]=math.radians(OPEN_DEG);append('open_gripper',.6,opened)
        cart('retreat',1.,layout.tray_xy,.105,layout.tray_yaw,OPEN_DEG)
        append('return_home',1.2,np.deg2rad(self.home_deg))
        append('settle_end',.6,q)
        self.duration=clock

    def sample(self,t):
        phase=self.phases[-1]
        for p in self.phases:
            if t<p.start+p.duration-1e-10:
                phase=p;break
        u=float(np.clip((t-phase.start)/phase.duration,0,1))
        smooth=u*u*u*(10+u*(-15+6*u))
        if phase.path is not None:
            index=smooth*(len(phase.path)-1);lo=min(int(index),len(phase.path)-2)
            q=phase.path[lo]+(phase.path[lo+1]-phase.path[lo])*(index-lo)
        else:q=phase.q0+(phase.q1-phase.q0)*smooth
        return q,phase.name,u

def sample_layout(assets,rng,tray_xy=None,tray_yaw=0.,fixed_cube=False,start_pose='folded'):
    for attempt in range(100):
        tray=np.array(tray_xy if tray_xy is not None else [rng.uniform(-.075,.075),rng.uniform(.235,.285)])
        tyaw=tray_yaw if tray_xy is not None else rng.uniform(-math.pi/3,math.pi/3)
        cube=np.array([-.045,.235] if fixed_cube else [rng.uniform(-.075,.075),rng.uniform(.22,.285)])
        if np.linalg.norm(cube-tray)<.075:continue
        layout=Layout(cube,0. if fixed_cube else rng.uniform(-math.pi/6,math.pi/6),tray,tyaw)
        try:return EpisodePlan(assets,layout,start_pose=start_pose)
        except ValueError:continue
    raise RuntimeError('Could not sample a reachable, separated cube/tray layout in 100 attempts')

class PickPlaceController:
    def __init__(self,scene,args):
        self.scene=scene;self.args=args;self.rng=np.random.default_rng(args.seed)
        self.episode_index=-1;self.results=[];self.holding=False;self.ever_grasped=False
        self.tray_xy=None;self.tray_yaw=0.;self.phase='';self.done=False
        self.start_episode(0.)

    def start_episode(self,t):
        self.episode_index+=1;self.start_time=t
        move_tray=self.tray_xy is None or (self.args.tray_every>0 and self.episode_index%self.args.tray_every==0)
        if self.args.replay_episode:
            source=json.loads(self.args.replay_episode.read_text())
            layout=Layout(np.array(source['cube_spawn_xy_m']),source['cube_spawn_yaw_rad'],
                          np.array(source['tray_xy_m']),source['tray_yaw_rad'])
            plan=EpisodePlan(self.scene.assets,layout,start_pose=self.args.start_pose)
        else:
            plan=sample_layout(self.scene.assets,self.rng,None if move_tray else self.tray_xy,
                               self.tray_yaw,self.args.fixed_cube,start_pose=self.args.start_pose)
        self.plan=plan;self.tray_xy=plan.layout.tray_xy.copy();self.tray_yaw=plan.layout.tray_yaw
        self.holding=False;self.ever_grasped=False;self.released=False;self.grasp_failure=None
        self.max_jaw_penetration=0.;self.max_environment_penetration=0.;self.premature_contact=False
        self.lifted=False;self.max_cube_height=WORK_SURFACE_Z+.015;self.max_grip_force=0.
        self.max_tcp_tracking_error=0.
        scene=self.scene;c=scene.c
        scene.tray.SetPos(scene.vec([*self.tray_xy,WORK_SURFACE_Z]))
        scene.tray.SetRot(c.QuatFromAngleZ(self.tray_yaw))
        scene.cube.SetFixed(False);scene.cube.EnableCollision(True)
        scene.cube.SetPos(scene.vec([*plan.layout.cube_xy,WORK_SURFACE_Z+.0152]))
        scene.cube.SetRot(c.QuatFromAngleZ(plan.layout.cube_yaw))
        scene.cube.SetPosDt(scene.vec([0,0,0]));scene.cube.SetAngVelParent(scene.vec([0,0,0]))
        scene.update_arm(t,np.rad2deg(plan.sample(0)[0]),reset=True)
        self.phase='settle_start'
        print(f'Auto episode {self.episode_index+1}/{self.args.episodes}: cube {np.round(plan.layout.cube_xy,3)}, tray {np.round(self.tray_xy,3)}')

    def update(self,t):
        elapsed=t-self.start_time
        if elapsed>=self.plan.duration-1e-8:return True
        q,self.phase,u=self.plan.sample(elapsed)
        self.scene.update_arm(t,np.rad2deg(q))
        if self.phase=='open_gripper' and u>.15:self.released=True
        return False

    def after_step(self,t):
        scene=self.scene
        contacts=scene.jaw_physics.observe()
        forces=[contacts.get(name,{}).get('normal_force_n',0.) for name in ['fixed_jaw','moving_jaw']]
        penetration=max([item['penetration_m'] for item in contacts.values()]+[0.])
        self.max_jaw_penetration=max(self.max_jaw_penetration,penetration)
        self.max_environment_penetration=max(self.max_environment_penetration,scene.jaw_physics.reporter.environment_penetration)
        self.max_grip_force=max(self.max_grip_force,*forces)
        if self.phase in ['approach_cube','descend_to_cube'] and max(forces)>.02:
            self.premature_contact=True
        if min(forces)>.02 and self.phase in ['close_gripper','lift_cube','transfer_to_tray','lower_into_tray']:
            self.ever_grasped=True
        position=np.array(scene.cube_pose()[:3]);tcp=scene.tool_world_pose()
        # Physics has advanced to t; joint_deg still describes the preceding
        # command tick. Compare the actual body with the trajectory at this t.
        planned=tool_pose(scene.assets,self.plan.sample(t-self.start_time)[0])
        self.max_tcp_tracking_error=max(self.max_tcp_tracking_error,float(np.linalg.norm(tcp[:3,3]-planned[:3,3])))
        self.max_cube_height=max(self.max_cube_height,float(position[2]))
        if self.ever_grasped and position[2]>WORK_SURFACE_Z+.06:self.lifted=True
        self.holding=bool(self.ever_grasped and not self.released and
            min(forces)>.01 and np.linalg.norm(position-tcp[:3,3])<.025 and position[2]>WORK_SURFACE_Z+.025)

    def next_action(self,t):
        return self.plan.sample(min(t-self.start_time+1/self.args.rate,self.plan.duration))[0]

    def finish_episode(self,t,complete=True):
        scene=self.scene
        cube=scene.body_transform(scene.cube);tray=scene.body_transform(scene.tray)
        relative=np.linalg.inv(tray)@cube
        corners=np.array([[x,y,z,1.] for x in [-.015,.015] for y in [-.015,.015] for z in [-.015,.015]])
        local=(relative@corners.T).T[:,:3]
        in_tray=bool(np.max(abs(local[:,:2]))<.0235 and abs(relative[2,3]-.017)<.002)
        quality_ok=not self.premature_contact and self.max_jaw_penetration<.0005 and self.max_environment_penetration<.0005 and self.max_tcp_tracking_error<.0005
        success=bool(complete and quality_ok and self.ever_grasped and self.lifted and self.released and not self.holding and in_tray)
        problems=[]
        if not self.ever_grasped:problems.append('No bilateral physical jaw contact')
        if not self.lifted:problems.append('Cube was not physically lifted above 6 cm')
        if self.premature_contact:problems.append('Jaw hit the cube before the closing phase')
        if max(self.max_jaw_penetration,self.max_environment_penetration)>=.0005:
            problems.append('Collision penetration exceeded 0.5 mm')
        if self.max_tcp_tracking_error>=.0005:problems.append('Prescribed wrist tracking error exceeded 0.5 mm')
        if not in_tray:problems.append('Cube did not settle inside the tray')
        self.grasp_failure='; '.join(problems) if problems else None
        result=dict(episode_index=self.episode_index,seed=self.args.seed,success=success,complete=complete,
            start_pose=self.plan.start_pose,home_joint_rad=self.plan.sample(0.)[0].tolist(),
            replay_layout_source=str(self.args.replay_episode) if self.args.replay_episode else None,
            grasp_mode=GRASP_MODE,grasped=self.ever_grasped,lifted=self.lifted,released=self.released,
            contact_quality_ok=quality_ok,premature_jaw_contact=self.premature_contact,
            max_jaw_penetration_m=self.max_jaw_penetration,max_environment_penetration_m=self.max_environment_penetration,
            max_cube_height_m=self.max_cube_height,max_grip_normal_force_n=self.max_grip_force,
            max_tcp_tracking_error_m=self.max_tcp_tracking_error,
            cube_in_tray=in_tray,grasp_failure=self.grasp_failure,
            cube_spawn_xy_m=self.plan.layout.cube_xy.tolist(),cube_spawn_yaw_rad=float(self.plan.layout.cube_yaw),
            tray_xy_m=self.tray_xy.tolist(),tray_yaw_rad=float(self.tray_yaw),
            cube_final_pose=scene.cube_pose(),duration_s=t-self.start_time,
            max_ik_error_m=max(r.position_error_m for r in self.plan.ik_results))
        self.results.append(result)
        print(f"Auto episode {self.episode_index+1}: {'SUCCESS' if success else 'FAILED'} (cube in tray: {in_tray})")
        return result
