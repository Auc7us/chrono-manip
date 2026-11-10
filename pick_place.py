"""Seeded episode planning with physical jaw contact and grasp validation."""
from dataclasses import dataclass
import hashlib
import json
import math
import numpy as np
from so101_ik import SO101IK, START_POSES_DEG, OPEN_DEG, GRASP_DEG, DOWN, tool_pose, forward_kinematics
from jaw_physics import GRASP_MODE
from scene_config import WORK_SURFACE_Z,tray_geometry

# Beyond ~6 mm at a 19 mm grasp height the moving jaw meets the cube more open, with
# its tip at the top edge; it perches there and the servo presses the cube down
# until it slips (seed 600, 20 grasps: 8 mm 13 pass, 11 mm 6 pass; at 17 mm, 20/20 both).
DESCENT_CENTERING_M=.006
GRASP_TCP_HEIGHT_M=.019
# Staged planner: top-down, the arm is nearly straight (>1 deg of joint motion per mm) past ~.245 m
# from the base at hover height. Farther out the gripper tilts back toward the base (cube distance m ->
# pitch deg), which keeps the arm bent out to ~.32 m at 30 deg of tilt.
REACH_PITCH=((.24,.32),(90.,60.))
# A 5-DOF arm turning its wrist about a tilted axis also tips the held cube; cap the turn at the tray
# so the cube tips at most this much (the cube still fits the tray at any yaw).
TRAY_TURN_TIP_DEG=8.
# DART noise envelope (start, end weight) per phase. The executed arm path is
# perturbed while recorded actions stay the clean plan, so demonstrations show
# the expert pulling back from small pose errors. Weights reach zero before
# jaw contact and release, so grasping and placement run on the clean path.
DART_ENVELOPE={'approach_cube':(0.,1.),'descend_to_cube':(1.,0.),'transfer_to_tray':(0.,1.),'lower_into_tray':(1.,0.)}
# Cautious planner: noise holds through each hover and fades out over the first part of the descent.
DART_ENVELOPE_CAUTIOUS={'approach_cube':(0.,1.),'hover_cube':(1.,1.),'descend_above_cube':(1.,0.),
    'transfer_to_tray':(0.,1.),'hover_tray':(1.,1.),'lower_into_tray':(1.,0.)}
# Aligned planner: noise also holds while the wrist turns to the tray just above it.
DART_ENVELOPE_ALIGNED={**DART_ENVELOPE_CAUTIOUS,'descend_above_tray':(1.,1.),'align_to_tray':(1.,1.)}
# Staged planner: the same, and noise also holds above the cube until the turn to the cube ends.
DART_ENVELOPE_STAGED={**DART_ENVELOPE_ALIGNED,'descend_above_cube':(1.,1.),'align_to_cube':(1.,0.)}
DART_ENVELOPES={'direct':DART_ENVELOPE,'cautious':DART_ENVELOPE_CAUTIOUS,'aligned':DART_ENVELOPE_ALIGNED,'staged':DART_ENVELOPE_STAGED}
DART_JOINT_WEIGHTS=np.array([1.,1.,1.,1.,2.,0.])

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
    def __init__(self,assets,layout,start_pose='folded',tray_style='reference',planner='direct'):
        self.layout=layout;self.phases=[];self.ik_results=[];self.planner=planner;self.variation={}
        self.tray_geometry=tray_geometry(tray_style)
        self.start_pose=start_pose;self.home_deg=START_POSES_DEG[start_pose].copy()
        self.assets=assets;self.ik=SO101IK(assets);self.clock=0.
        if planner in ['cautious','aligned','staged']:self.plan_cautious()
        else:self.plan_direct()
        self.duration=self.clock

    def append(self,name,duration,target,path=None):
        self.phases.append(Phase(name,self.clock,duration,self.q.copy(),target.copy(),path))
        self.clock+=duration;self.q=target.copy()

    def cart(self,name,duration,xy,z,yaw,jaw,steps=20,pitch_deg=None,bulge=0.):
        """Straight TCP path (optionally bowed sideways by `bulge` m) with yaw, and for the cautious
        planner the gripper pitch, interpolated from the current pose; pitch None keeps it vertical."""
        start=tool_pose(self.assets,self.q);target=np.array([*xy,WORK_SURFACE_Z+z]);path=[self.q.copy()]
        initial_yaw=math.atan2(start[1,1],start[0,1])
        yaw_delta=(yaw-initial_yaw+math.pi)%(2*math.pi)-math.pi
        initial_pitch=math.degrees(math.asin(float(np.clip(-start[2,0],-1,1))))
        side=np.array([-(target-start[:3,3])[1],(target-start[:3,3])[0],0.]);side/=max(np.linalg.norm(side),1e-9)
        fractions=[1.] if name=='approach_cube' and self.planner=='direct' else np.linspace(.05,1.,steps)
        for fraction in fractions:
            position=start[:3,3]+fraction*(target-start[:3,3])+bulge*4*fraction*(1-fraction)*side
            approach=DOWN
            if pitch_deg is not None:
                # A 5-DOF arm can only pitch within the vertical plane through its base.
                pitch=math.radians(initial_pitch+fraction*(pitch_deg-initial_pitch));heading=position[:2]-self.base_xy
                heading/=np.linalg.norm(heading);approach=np.r_[heading*math.cos(pitch),-math.sin(pitch)]
            result=self.ik.solve(position,path[-1],yaw=initial_yaw+fraction*yaw_delta,jaw_deg=jaw,approach=approach)
            if not result.success:
                raise ValueError(f'{name} unreachable: {result.position_error_m*1000:.2f} mm, {result.approach_error_deg:.2f} deg')
            # Near the base the elbow folds to its limit and IK can switch branch mid-path.
            if self.planner!='direct' and np.rad2deg(np.abs(result.q[:5]-path[-1][:5])).max()>12:
                raise ValueError(f'{name} switches IK branch')
            self.ik_results.append(result);path.append(result.q)
        self.append(name,duration,result.q,np.array(path) if len(path)>2 else None)

    def plan_direct(self):
        layout=self.layout;self.q=np.deg2rad(self.home_deg)
        self.append('settle_start',.4,self.q)
        # Descend with the cube ~7 mm from the fixed teeth instead of ~1 mm;
        # closing slides it onto the fixed jaw, where the carry expects it.
        grasp_xy=layout.cube_xy-DESCENT_CENTERING_M*np.array([math.cos(layout.cube_yaw),math.sin(layout.cube_yaw)])
        self.cart('approach_cube',1.2,grasp_xy,.105,layout.cube_yaw,OPEN_DEG)
        self.cart('descend_to_cube',1.,grasp_xy,GRASP_TCP_HEIGHT_M,layout.cube_yaw,OPEN_DEG)
        closed=self.q.copy();closed[5]=math.radians(GRASP_DEG);self.append('close_gripper',.6,closed)
        self.cart('lift_cube',1.,grasp_xy,.105,layout.cube_yaw,GRASP_DEG)
        self.cart('transfer_to_tray',1.4,layout.tray_xy,.105,layout.tray_yaw,GRASP_DEG)
        # Release near the plate, raising the wrist for taller tray walls so
        # the jaws remain clear throughout opening.
        self.cart('lower_into_tray',1.,layout.tray_xy,self.tray_geometry['release_tcp_height_above_mat_m'],layout.tray_yaw,GRASP_DEG)
        opened=self.q.copy();opened[5]=math.radians(OPEN_DEG);self.append('open_gripper',.6,opened)
        self.cart('retreat',1.,layout.tray_xy,.105,layout.tray_yaw,OPEN_DEG)
        self.append('return_home',1.2,np.deg2rad(self.home_deg))
        self.append('settle_end',.6,self.q)

    def plan_cautious(self):
        """Up, across, pause, then down: hover above each target fully aligned before descending.
        Speed, hover/carry heights, a sideways bow in the carry and the start pose vary per layout,
        seeded from the layout itself so replays and resumes rebuild the identical plan.
        The aligned planner carries without turning the wrist, lowers to just above the tray, turns
        to the tray there (in view of the wrist camera) and then lowers slowly to the release. The staged
        planner also approaches the cube without turning the wrist and turns to the cube just above it;
        past top-down reach it tilts the gripper and holds that tilt until the cube is released."""
        layout=self.layout
        key=np.r_[layout.cube_xy,layout.cube_yaw,layout.tray_xy,layout.tray_yaw].astype(float).tobytes()
        rng=np.random.default_rng(int.from_bytes(hashlib.sha256(key).digest()[:8],'little'))
        slow=rng.uniform(.8,1.25);hover=rng.uniform(.08,.13);carry=rng.uniform(.08,.13);bulge=rng.uniform(-.03,.03)
        dwell=rng.uniform(.3,.5);self.home_deg[:5]+=rng.uniform(-4,4,5)
        self.base_xy=forward_kinematics(self.assets,self.home_deg)[0][:2,3]
        natural=lambda xy:math.atan2(*(xy-self.base_xy)[::-1])-math.pi/2
        # The cube and tray are square: of the four equivalent headings take the one nearest the
        # wrist's unrolled heading, so the wrist never turns more than 45 deg for either.
        grasp_yaw=natural(layout.cube_xy)+(layout.cube_yaw-natural(layout.cube_xy)+math.pi/4)%(math.pi/2)-math.pi/4
        offset=grasp_yaw-natural(layout.cube_xy)
        place_yaw=natural(layout.tray_xy)+offset+(layout.tray_yaw-natural(layout.tray_xy)-offset+math.pi/4)%(math.pi/2)-math.pi/4
        grasp_xy=layout.cube_xy-DESCENT_CENTERING_M*np.array([math.cos(grasp_yaw),math.sin(grasp_yaw)])
        release=self.tray_geometry['release_tcp_height_above_mat_m']
        staged=self.planner=='staged';approach_yaw=natural(layout.cube_xy) if staged else grasp_yaw
        pitch=float(np.interp(np.linalg.norm(grasp_xy-self.base_xy),*REACH_PITCH)) if staged else 90.
        tilt=pitch if staged else None
        if staged and pitch<90.:
            carry_yaw=natural(layout.tray_xy)+offset
            turn=2*math.asin(min(1.,math.sin(math.radians(TRAY_TURN_TIP_DEG)/2)/math.cos(math.radians(pitch))))
            place_yaw=carry_yaw+float(np.clip(place_yaw-carry_yaw,-turn,turn))
        home=tool_pose(self.assets,np.deg2rad(self.home_deg));home_pitch=math.degrees(math.asin(float(-home[2,0])))
        home_yaw=math.atan2(home[1,1],home[0,1])
        for lower in [0.,.02,.04,.06]:
            # Far targets are only reachable top-down from lower; keep >= 7 cm.
            h=max(hover-lower,.07);c=max(carry-lower,.07);travel=np.linalg.norm(layout.tray_xy-grasp_xy)
            self.phases=[];self.ik_results=[];self.clock=0.;self.q=np.deg2rad(self.home_deg)
            try:
                self.append('settle_start',.4,self.q)
                self.cart('approach_cube',(.9+np.linalg.norm(grasp_xy-home[:2,3])/.25)*slow,grasp_xy,h,approach_yaw,OPEN_DEG,40 if staged else 25,pitch_deg=pitch)
                self.append('hover_cube',dwell,self.q)
                self.cart('descend_above_cube',.6*slow,grasp_xy,GRASP_TCP_HEIGHT_M+.035,approach_yaw,OPEN_DEG,pitch_deg=tilt)
                if staged:
                    self.cart('align_to_cube',(.4+.8*abs(grasp_yaw-approach_yaw)/(math.pi/4))*slow,grasp_xy,GRASP_TCP_HEIGHT_M+.035,grasp_yaw,OPEN_DEG,
                              pitch_deg=tilt)
                self.cart('descend_to_cube',.8*slow,grasp_xy,GRASP_TCP_HEIGHT_M,grasp_yaw,OPEN_DEG,pitch_deg=tilt)
                closed=self.q.copy();closed[5]=math.radians(GRASP_DEG);self.append('close_gripper',.6,closed)
                if self.planner in ['aligned','staged']:
                    # The cube slowly pivots in the grip (~3-4 deg/s), and past ~20 deg it lands on the rim, so
                    # the lift and carry are quicker and there are no dwells: the cube is held only ~1 s longer
                    # than in the cautious plan, all of it spent turning and lowering just above the tray.
                    carry_yaw=natural(layout.tray_xy)+offset
                    self.cart('lift_cube',.8*slow,grasp_xy,c,grasp_yaw,GRASP_DEG,pitch_deg=tilt)
                    self.cart('transfer_to_tray',(.6+travel/.3)*slow,layout.tray_xy,c,carry_yaw,GRASP_DEG,30,pitch_deg=tilt,bulge=bulge)
                    self.cart('descend_above_tray',.6*slow,layout.tray_xy,release+.035,carry_yaw,GRASP_DEG,pitch_deg=tilt)
                    self.cart('align_to_tray',(.25+.55*abs(place_yaw-carry_yaw)/(math.pi/4))*slow,layout.tray_xy,release+.035,place_yaw,GRASP_DEG,
                              pitch_deg=tilt)
                    self.cart('lower_into_tray',1.2*slow,layout.tray_xy,release,place_yaw,GRASP_DEG,pitch_deg=tilt)
                else:
                    self.cart('lift_cube',1.*slow,grasp_xy,c,grasp_yaw,GRASP_DEG)
                    self.cart('transfer_to_tray',(.8+travel/.25)*slow,layout.tray_xy,c,place_yaw,GRASP_DEG,30,bulge=bulge)
                    self.append('hover_tray',dwell*.75,self.q)
                    self.cart('lower_into_tray',1.*slow,layout.tray_xy,release,place_yaw,GRASP_DEG)
                opened=self.q.copy();opened[5]=math.radians(OPEN_DEG);self.append('open_gripper',.6,opened)
                self.cart('retreat',.8*slow,layout.tray_xy,h,place_yaw,OPEN_DEG,pitch_deg=tilt)
                self.cart('return_home',(.9+np.linalg.norm(layout.tray_xy-home[:2,3])/.25)*slow,home[:2,3],home[2,3]-WORK_SURFACE_Z,
                          home_yaw,OPEN_DEG,25,pitch_deg=home_pitch)
                rest=np.deg2rad(self.home_deg);rest[5]=self.q[5];self.append('settle_home',.3,rest)
                self.append('settle_end',.6,np.deg2rad(self.home_deg))
            except ValueError:
                if h<=.07 and c<=.07:raise
                continue
            break
        self.variation=dict(speed_scale=1/slow,hover_tcp_height_m=h,carry_tcp_height_m=c,carry_bow_m=bulge,dwell_s=dwell,
            start_joint_deg=self.home_deg.tolist(),grasp_yaw_rad=grasp_yaw,place_yaw_rad=place_yaw,grasp_pitch_deg=pitch)

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

def sample_layout(assets,rng,tray_xy=None,tray_yaw=0.,fixed_cube=False,start_pose='folded',tray_style='reference',planner='direct'):
    for attempt in range(100):
        if planner!='direct':
            # Anywhere on the mat the arm can reach top-down; any orientation (both are square).
            tray=np.array(tray_xy if tray_xy is not None else [rng.uniform(-.19,.19),rng.uniform(.19,.34)])
            tyaw=tray_yaw if tray_xy is not None else rng.uniform(-math.pi/4,math.pi/4)
            # The staged planner tilts the gripper for far cubes, so its cubes cover the mat out to that reach.
            far=planner=='staged' and not fixed_cube
            cube=np.array([-.045,.235] if fixed_cube else [rng.uniform(-.21,.21),rng.uniform(.175,.42)] if far else [rng.uniform(-.2,.2),rng.uniform(.175,.34)])
            cyaw=0. if fixed_cube else rng.uniform(-math.pi/4,math.pi/4)
            if np.linalg.norm(cube-tray)<.08:continue
            if far and np.linalg.norm(cube-forward_kinematics(assets,START_POSES_DEG[start_pose])[0][:2,3])>REACH_PITCH[0][1]:continue
        else:
            tray=np.array(tray_xy if tray_xy is not None else [rng.uniform(-.075,.075),rng.uniform(.235,.285)])
            tyaw=tray_yaw if tray_xy is not None else rng.uniform(-math.pi/3,math.pi/3)
            cube=np.array([-.045,.235] if fixed_cube else [rng.uniform(-.075,.075),rng.uniform(.22,.285)])
            if np.linalg.norm(cube-tray)<.075:continue
            cyaw=0. if fixed_cube else rng.uniform(-math.pi/6,math.pi/6)
        layout=Layout(cube,cyaw,tray,tyaw)
        try:return EpisodePlan(assets,layout,start_pose=start_pose,tray_style=tray_style,planner=planner)
        except ValueError:continue
    raise RuntimeError('Could not sample a reachable, separated cube/tray layout in 100 attempts')

class PickPlaceController:
    def __init__(self,scene,args,resume=None):
        self.scene=scene;self.args=args;self.rng=np.random.default_rng(args.seed)
        self.episode_index=-1;self.results=[];self.holding=False;self.ever_grasped=False
        self.tray_xy=None;self.tray_yaw=0.;self.phase='';self.done=False
        if resume:self.restore_resume(resume)
        else:self.start_episode(0.)

    def make_plan(self,index):
        move_tray=self.tray_xy is None or (self.args.tray_every>0 and index%self.args.tray_every==0)
        if self.args.replay_episode:
            source=json.loads(self.args.replay_episode.read_text())
            layout=Layout(np.array(source['cube_spawn_xy_m']),source['cube_spawn_yaw_rad'],
                          np.array(source['tray_xy_m']),source['tray_yaw_rad'])
            plan=EpisodePlan(self.scene.assets,layout,start_pose=self.args.start_pose,tray_style=self.args.tray_style,planner=self.args.planner)
        else:
            plan=sample_layout(self.scene.assets,self.rng,None if move_tray else self.tray_xy,
                               self.tray_yaw,self.args.fixed_cube,start_pose=self.args.start_pose,tray_style=self.args.tray_style,planner=self.args.planner)
        return plan

    def restore_resume(self,state):
        next_index=state['next_index'];self.results=list(state['completed_infos']);checkpoint=state['checkpoint'];layout=None
        if checkpoint and checkpoint.get('seed')==self.args.seed and checkpoint.get('episode_index') in [next_index-1,next_index]:
            self.rng.bit_generator.state=checkpoint['rng_state'];saved=checkpoint['layout']
            self.tray_xy=np.array(saved['tray_xy']);self.tray_yaw=saved['tray_yaw']
            if checkpoint['episode_index']==next_index:
                layout=Layout(np.array(saved['cube_xy']),saved['cube_yaw'],self.tray_xy.copy(),self.tray_yaw)
            print(f'Resuming after {next_index} completed episodes using saved RNG/layout checkpoint.',flush=True)
        else:
            print(f'Rebuilding seeded layout history for {next_index} completed episodes (legacy recording).',flush=True)
            for index,info in enumerate(state['completed_infos']):
                plan=self.make_plan(index)
                expected=np.r_[info['cube_spawn_xy_m'],info['cube_spawn_yaw_rad'],info['tray_xy_m'],info['tray_yaw_rad']]
                actual=np.r_[plan.layout.cube_xy,plan.layout.cube_yaw,plan.layout.tray_xy,plan.layout.tray_yaw]
                if not np.allclose(actual,expected,atol=1e-9,rtol=0):raise ValueError(f'Saved layout history differs at episode {index}; refusing an inconsistent resume')
                self.tray_xy=plan.layout.tray_xy.copy();self.tray_yaw=plan.layout.tray_yaw
        self.episode_index=next_index-1
        self.start_episode(0.,layout=layout)

    def start_episode(self,t,layout=None):
        self.episode_index+=1;self.start_time=t
        plan=self.make_plan(self.episode_index) if layout is None else EpisodePlan(self.scene.assets,layout,start_pose=self.args.start_pose,tray_style=self.args.tray_style,planner=self.args.planner)
        self.plan=plan;self.tray_xy=plan.layout.tray_xy.copy();self.tray_yaw=plan.layout.tray_yaw
        self.holding=False;self.ever_grasped=False;self.released=False;self.grasp_failure=None
        self.max_jaw_penetration=0.;self.max_environment_penetration=0.;self.premature_contact=False
        self.lifted=False;self.max_cube_height=WORK_SURFACE_Z+.015;self.max_grip_force=0.
        self.max_tcp_tracking_error=0.
        # Separate stream per episode: layouts and resume stay independent of noise.
        noise=np.random.default_rng([self.args.seed,self.episode_index])
        self.dart_amplitude_deg=float(noise.uniform(0.,getattr(self.args,'dart_noise_deg',0.)))
        self.dart_frequencies=noise.uniform(.2,1.,(6,3));self.dart_phases=noise.uniform(0.,2*math.pi,(6,3))
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

    def command(self,elapsed):
        q,phase,u=self.plan.sample(elapsed)
        weights=DART_ENVELOPES[self.plan.planner].get(phase)
        if weights and self.dart_amplitude_deg:
            s=u*u*(3-2*u);envelope=weights[0]+(weights[1]-weights[0])*s
            # Three slow sinusoids per joint, scaled to unit RMS.
            wave=np.sin(2*math.pi*self.dart_frequencies*elapsed+self.dart_phases).sum(axis=1)*math.sqrt(2/3)
            q=q+np.radians(self.dart_amplitude_deg*envelope*DART_JOINT_WEIGHTS*wave)
        return q,phase,u

    def update(self,t):
        elapsed=t-self.start_time
        if elapsed>=self.plan.duration-1e-8:return True
        q,self.phase,u=self.command(elapsed)
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
        if self.phase in ['approach_cube','hover_cube','descend_above_cube','align_to_cube','descend_to_cube'] and max(forces)>.02:
            self.premature_contact=True
        if min(forces)>.02 and self.phase in ['close_gripper','lift_cube','transfer_to_tray','hover_tray','descend_above_tray','align_to_tray','lower_into_tray']:
            self.ever_grasped=True
        position=np.array(scene.cube_pose()[:3]);tcp=scene.tool_world_pose()
        # Physics has advanced to t; joint_deg still describes the preceding
        # command tick. Compare the actual body with the trajectory at this t.
        planned=tool_pose(scene.assets,self.command(t-self.start_time)[0])
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
            start_pose=self.plan.start_pose,home_joint_rad=self.plan.sample(0.)[0].tolist(),planner=self.plan.planner,plan_variation=self.plan.variation,
            replay_layout_source=str(self.args.replay_episode) if self.args.replay_episode else None,
            grasp_mode=GRASP_MODE,grasped=self.ever_grasped,lifted=self.lifted,released=self.released,
            contact_quality_ok=quality_ok,premature_jaw_contact=self.premature_contact,
            max_jaw_penetration_m=self.max_jaw_penetration,max_environment_penetration_m=self.max_environment_penetration,
            max_cube_height_m=self.max_cube_height,max_grip_normal_force_n=self.max_grip_force,
            max_tcp_tracking_error_m=self.max_tcp_tracking_error,dart_noise_deg=self.dart_amplitude_deg,
            cube_in_tray=in_tray,grasp_failure=self.grasp_failure,
            tray_geometry=scene.tray_geometry,
            cube_spawn_xy_m=self.plan.layout.cube_xy.tolist(),cube_spawn_yaw_rad=float(self.plan.layout.cube_yaw),
            tray_xy_m=self.tray_xy.tolist(),tray_yaw_rad=float(self.tray_yaw),
            cube_final_pose=scene.cube_pose(),duration_s=t-self.start_time,
            max_ik_error_m=max(r.position_error_m for r in self.plan.ik_results))
        self.results.append(result)
        print(f"Auto episode {self.episode_index+1}: {'SUCCESS' if success else 'FAILED'} (cube in tray: {in_tray})")
        return result
