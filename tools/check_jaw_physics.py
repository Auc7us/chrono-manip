"""Run and log a CPU-only physical grasp; execute with the demo's Chrono environment."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode',type=Path);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=42);parser.add_argument('--episodes',type=int,default=1)
    parser.add_argument('--start-pose',choices=['folded','side'],default='folded')
    parser.add_argument('--probe-fixed-jaw',action='store_true',help='deliberately reproduce the old tooth overlap and require a physical cube response')
    parser.add_argument('--with-sensors',action='store_true',help='also profile headless wrist/global camera rendering')
    parser.add_argument('--rate',type=int,default=15)
    options=parser.parse_args()
    from demo_attempt_codex import Scene,parse_args
    from pick_place import PickPlaceController
    from so101_ik import tool_pose
    from scene_config import WORK_SURFACE_Z
    sys.argv=['check','--auto','--headless','--no-sensors','--no-realtime','--seed',str(options.seed),
              '--episodes',str(options.episodes),'--start-pose',options.start_pose,'--rate',str(options.rate)]
    if options.with_sensors:sys.argv.remove('--no-sensors')
    if options.episode:sys.argv+=['--replay-episode',str(options.episode)]
    args=parse_args()
    import pychrono.core as chrono
    assets=json.loads((ROOT/'assets/scene_assets.json').read_text())
    scene=Scene(chrono,assets,args);controller=PickPlaceController(scene,args);rows=[];t=0.;tick=0
    manager=None;sensors={};timings={}
    if options.with_sensors:
        import pychrono.sensor as sensor
        from demo_attempt_codex import configure_sensors
        manager,sensors=configure_sensors(scene,sensor)
    if options.probe_fixed_jaw:
        phase=next(p for p in controller.plan.phases if p.name=='descend_to_cube')
        q=phase.q1
        scene.update_arm(0.,np.rad2deg(q),reset=True)
        direction=tool_pose(assets,q)[:3,1]
        initial=np.r_[controller.plan.layout.cube_xy,WORK_SURFACE_Z+.015]-.0035*direction
        scene.cube.SetPos(scene.vec(initial))
        force=0.;penetration=0.
        for i in range(100):
            scene.update_arm(t,np.rad2deg(q));scene.system.DoStepDynamics(.001);t=scene.system.GetChTime()
            item=scene.jaw_physics.observe().get('fixed_jaw',{})
            force=max(force,item.get('normal_force_n',0.));penetration=max(penetration,item.get('penetration_m',0.))
        displacement=float(np.dot(np.array(scene.cube_pose()[:3])-initial,direction))
        result=dict(fixed_jaw_peak_force_n=force,initial_overlap_m=penetration,cube_displacement_away_from_teeth_m=displacement)
        assert force>.01 and penetration>.001 and displacement>.001,result
        options.output.mkdir(parents=True,exist_ok=True)
        (options.output/'probe.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2));return
    while True:
        control_start=time.perf_counter()
        if controller.update(t):
            controller.finish_episode(t)
            if controller.episode_index+1>=args.episodes:break
            controller.start_episode(t)
        control_seconds=time.perf_counter()-control_start
        phase=controller.phase
        record=timings.setdefault(phase,dict(steps=0,control_seconds=0.,dynamics_seconds=0.,
            contact_reporting_seconds=0.,sensor_seconds=0.,collision_seconds=0.,solve_seconds=0.,
            setup_seconds=0.,update_seconds=0.,contacts_sum=0,contacts_peak=0))
        record['control_seconds']+=control_seconds
        if manager:
            before=time.perf_counter();manager.Update();record['sensor_seconds']+=time.perf_counter()-before
        if tick%10==0:
            contacts=scene.jaw_physics.contacts
            rows.append(dict(time=t,episode=controller.episode_index,phase=controller.phase,
                state=np.deg2rad(scene.joint_deg),target=controller.plan.sample(t-controller.start_time)[0],
                cube_pose=scene.cube_pose(),tcp_pose=scene.tool_world_pose(),
                jaw_force=[contacts.get(k,{}).get('normal_force_n',0.) for k in ['fixed_jaw','moving_jaw']],
                jaw_torque=scene.jaw_physics.torque,
                penetration=[contacts.get(k,{}).get('penetration_m',0.) for k in ['fixed_jaw','moving_jaw']]))
        before=time.perf_counter();scene.system.DoStepDynamics(.001)
        record['dynamics_seconds']+=time.perf_counter()-before
        t=scene.system.GetChTime();before=time.perf_counter();controller.after_step(t)
        record['contact_reporting_seconds']+=time.perf_counter()-before
        record['steps']+=1
        for key,getter in [('collision_seconds','GetTimerCollision'),('solve_seconds','GetTimerLSsolve'),
                           ('setup_seconds','GetTimerLSsetup'),('update_seconds','GetTimerUpdate')]:
            record[key]+=getattr(scene.system,getter)()
        count=scene.system.GetNumContacts();record['contacts_sum']+=count
        record['contacts_peak']=max(record['contacts_peak'],count);tick+=1
    options.output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(options.output/'physics.npz',**{k:np.array([row[k] for row in rows]) for k in rows[0]})
    (options.output/'results.json').write_text(json.dumps(controller.results,indent=2)+'\n')
    for record in timings.values():
        record['simulated_seconds']=record['steps']*.001
        record['dynamics_rtf']=record['simulated_seconds']/record['dynamics_seconds']
        record['measured_loop_rtf']=record['simulated_seconds']/sum(record[k] for k in
            ['control_seconds','dynamics_seconds','contact_reporting_seconds','sensor_seconds'])
        record['mean_contacts']=record['contacts_sum']/record['steps']
    profile=dict(with_sensors=options.with_sensors,rate=args.rate,width=args.width,samples=args.samples,
        step_s=.001,phases=timings,notes='RTF = simulated seconds / wall seconds. Excludes startup, output encoding, and diagnostic sample logging.')
    (options.output/'profile.json').write_text(json.dumps(profile,indent=2)+'\n')
    print(json.dumps(controller.results,indent=2))
    print('Per-phase timing:',options.output/'profile.json')
    sensors.clear()
    if manager:del manager

if __name__=='__main__':main()
