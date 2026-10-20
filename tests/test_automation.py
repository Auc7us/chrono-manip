"""CPU checks; run from the demo folder: python3 -m unittest discover -s tests."""
import json
from pathlib import Path
import unittest
import numpy as np
from so101_ik import HOME_DEG, START_POSES_DEG, LIMITS_DEG, SO101IK, tool_pose, forward_kinematics
from pick_place import sample_layout
from episode_data import capture_synchronized
from scene_config import WIDTH,WORK_SURFACE_Z,SHEET_SIZE,SHEET_CENTER

ASSETS=json.loads((Path(__file__).resolve().parents[1]/'assets/scene_assets.json').read_text())

class AutomationTests(unittest.TestCase):
    def test_detailed_jaws_clear_cube_before_closing(self):
        from so101_ik import CLOSING
        root=Path(__file__).resolve().parents[1]
        geometry=json.loads((root/'assets/jaw_collision.json').read_text())
        plan=sample_layout(ASSETS,np.random.default_rng(42))
        phase=next(p for p in plan.phases if p.name=='descend_to_cube')
        frames=forward_kinematics(ASSETS,np.rad2deg(phase.q1))
        yaw=plan.layout.cube_yaw;c,s=np.cos(yaw),np.sin(yaw)
        rotation=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
        center=np.r_[plan.layout.cube_xy,WORK_SURFACE_Z+.015]
        def intersects(vertices,faces):
            triangles=vertices[faces]
            edges=np.roll(triangles,-1,axis=1)-triangles
            axes=np.concatenate((np.broadcast_to(np.eye(3),(len(faces),3,3)),
                np.cross(edges[:,:,None,:],np.eye(3)[None,None,:,:]).reshape(-1,9,3),
                np.cross(edges[:,0],edges[:,1])[:,None,:]),axis=1)
            projections=np.einsum('nvk,nak->nva',triangles,axes)
            radius=.015*abs(axes).sum(2)
            return np.all((projections.min(1)<=radius)&(projections.max(1)>=-radius),axis=1).any()
        for name,part in geometry['parts'].items():
            lines=(root/'assets/meshes'/part['file']).read_text().splitlines()
            vertices=np.array([[float(x) for x in line.split()[1:4]] for line in lines if line.startswith('v ')])
            faces=np.array([[int(x)-1 for x in line.split()[1:4]] for line in lines if line.startswith('f ')])
            self.assertGreater(len(faces),9000)
            self.assertGreaterEqual(len(faces),part['source_triangles']-2)
            f=frames[part['link']];world=vertices@f[:3,:3].T+f[:3,3]
            self.assertFalse(intersects((world-center)@rotation,faces),name)
            if name=='fixed_jaw':
                # Regression: the old TCP placed the same complete tooth mesh
                # 3.5 mm farther into the cube. No source recording is required.
                old_world=world+.0035*(frames[5][:3,:3]@CLOSING)
                self.assertTrue(intersects((old_world-center)@rotation,faces))

    def test_lowering_and_lifting_are_vertical(self):
        plan=sample_layout(ASSETS,np.random.default_rng(42))
        for phase in plan.phases:
            if phase.name not in ['descend_to_cube','lift_cube','lower_into_tray','retreat']:continue
            xy=tool_pose(ASSETS,phase.q1)[:2,3]
            for t in np.linspace(phase.start,phase.start+phase.duration,31):
                actual=tool_pose(ASSETS,plan.sample(t)[0])[:2,3]
                np.testing.assert_allclose(actual,xy,atol=.0001)

    def test_start_pose_selection_and_return(self):
        layouts=[]
        for name,home in START_POSES_DEG.items():
            for seed in range(5):
                plan=sample_layout(ASSETS,np.random.default_rng(seed),start_pose=name)
                np.testing.assert_allclose(plan.sample(0.)[0],np.deg2rad(home))
                np.testing.assert_allclose(plan.sample(plan.duration)[0],np.deg2rad(home))
                self.assertTrue(all(result.success for result in plan.ik_results))
                if seed==0:layouts.append(plan.layout)
        np.testing.assert_array_equal(layouts[0].cube_xy,layouts[1].cube_xy)
        np.testing.assert_array_equal(layouts[0].tray_xy,layouts[1].tray_xy)
        self.assertGreater(np.linalg.norm(tool_pose(ASSETS,np.deg2rad(START_POSES_DEG['side']))[:2,3]),.3)

    def test_global_camera_configurations(self):
        from demo_attempt_codex import WIDTH, HEIGHT, DEPTH, global_camera_mount
        name,top=global_camera_mount('top-down')
        self.assertEqual(name,'top_down')
        np.testing.assert_allclose(top[:3,3],[0.,DEPTH/2,HEIGHT-.007])
        np.testing.assert_allclose(top[:3,0],[0.,0.,-1.])
        name,right=global_camera_mount('right-45')
        self.assertEqual(name,'right_45')
        np.testing.assert_allclose(right[:3,:3].T@right[:3,:3],np.eye(3),atol=1e-12)
        self.assertGreater(right[2,2],0.,'Image up should point toward the ceiling')
        lens=right[:3,3]+right[:3,0]*ASSETS['wrist_camera']['optical_offset_m']
        np.testing.assert_allclose(lens,[WIDTH/2-.040,DEPTH/2,HEIGHT-.020],atol=1e-12)
        np.testing.assert_allclose(right[:3,0],[-2**-.5,0.,-2**-.5],atol=1e-12)
        # All randomized cube/tray footprints fit in the nominal wide-angle view.
        targets=np.array([[x,y,.03] for x in [-.112,.112] for y in [.198,.322]])
        from reference_camera import project_world
        pixels=project_world(targets)
        self.assertTrue(((pixels[:,0]>0)&(pixels[:,0]<640)).all())
        self.assertTrue(((pixels[:,1]>0)&(pixels[:,1]<480)).all())

    def test_home_is_folded_back_with_wrist_down(self):
        q=np.deg2rad(HOME_DEG);tcp=tool_pose(ASSETS,q)
        self.assertLess(tcp[2,0],-.995)
        self.assertGreater(tcp[2,3],.075)
        self.assertLess(np.linalg.norm(tcp[:2,3]-[0.,.054]),.13)
        frames=forward_kinematics(ASSETS,HOME_DEG)
        pivots=[frames[i][:3,:3]@np.array(ASSETS['joints'][i]['pivot_m'])+frames[i][:3,3]
                for i in [1,2,3]]
        upper=pivots[1]-pivots[0];forearm=pivots[2]-pivots[1]
        cosine=np.dot(upper,forearm)/(np.linalg.norm(upper)*np.linalg.norm(forearm))
        self.assertLess(cosine,-.2,'Forearm should fold back toward the shoulder')

    def test_home_clears_overhead_spawn_region(self):
        from demo_attempt_codex import HEIGHT, DEPTH
        root=Path(__file__).resolve().parents[1]
        camera=np.array([0.,DEPTH/2,HEIGHT-.007-ASSETS['wrist_camera']['optical_offset_m']])
        frames=forward_kinematics(ASSETS,HOME_DEG)
        # Includes all cube/tray spawns, with room for their rotated footprints.
        region_lo=np.array([-.112,.198]);region_hi=np.array([.112,.322])
        parts=ASSETS['meshes']+[dict(file='wrist_camera_mount.obj',link=5)]
        for part in parts:
            vertices=np.array([[float(x) for x in line.split()[1:4]]
                for line in (root/'assets/meshes'/part['file']).read_text().splitlines()
                if line.startswith('v ')])
            frame=frames[part['link']]
            world=vertices@frame[:3,:3].T+frame[:3,3]
            self.assertTrue((world[:,2]<camera[2]).all())
            # Perspective projection onto the cube's top plane from the actual
            # ceiling lens center; bounding boxes conservatively cover triangles.
            projected=camera+(world-camera)*((.03-camera[2])/(world[:,2]-camera[2]))[:,None]
            lo=projected[:,:2].min(0);hi=projected[:,:2].max(0)
            overlap=np.all(hi>region_lo) and np.all(lo<region_hi)
            self.assertFalse(overlap,f'{part["file"]} hides the spawn region at home')
            self.assertTrue((abs(world[:,0])<WIDTH/2).all(),part['file'])

    def test_random_plans_reach_targets_and_stay_continuous(self):
        rng=np.random.default_rng(523)
        for _ in range(20):
            plan=sample_layout(ASSETS,rng)
            for xy,half_diagonal in [(plan.layout.cube_xy,.015*np.sqrt(2)),(plan.layout.tray_xy,.025*np.sqrt(2))]:
                self.assertTrue((abs(xy-np.array(SHEET_CENTER[:2]))+half_diagonal<np.array(SHEET_SIZE[:2])/2).all())
            self.assertGreaterEqual(np.linalg.norm(plan.layout.cube_xy-plan.layout.tray_xy),.075)
            self.assertAlmostEqual(plan.duration,10.)
            for result in plan.ik_results:
                self.assertTrue(result.success)
                self.assertLess(result.position_error_m,.0015)
                self.assertTrue((np.rad2deg(result.q)>=LIMITS_DEG[:,0]).all())
                self.assertTrue((np.rad2deg(result.q)<=LIMITS_DEG[:,1]).all())
            for phase in plan.phases[1:]:
                before=plan.sample(phase.start-1e-7)[0]
                after=plan.sample(phase.start+1e-7)[0]
                np.testing.assert_allclose(before,after,atol=1e-8)
            np.testing.assert_allclose(plan.sample(10.)[0],np.deg2rad(HOME_DEG))
            for name,xy,z in [('descend_to_cube',plan.layout.cube_xy,.019),
                              ('lower_into_tray',plan.layout.tray_xy,.034)]:
                phase=next(p for p in plan.phases if p.name==name)
                tcp=tool_pose(ASSETS,phase.q1)
                np.testing.assert_allclose(tcp[:3,3],[*xy,WORK_SURFACE_Z+z],atol=.0015)
                self.assertLess(tcp[2,0],-.997)

    def test_seed_and_fixed_tray(self):
        a=sample_layout(ASSETS,np.random.default_rng(42))
        b=sample_layout(ASSETS,np.random.default_rng(42))
        np.testing.assert_array_equal(a.layout.cube_xy,b.layout.cube_xy)
        np.testing.assert_array_equal(a.layout.tray_xy,b.layout.tray_xy)
        c=sample_layout(ASSETS,np.random.default_rng(43),a.layout.tray_xy,a.layout.tray_yaw)
        np.testing.assert_array_equal(c.layout.tray_xy,a.layout.tray_xy)
        self.assertEqual(c.layout.tray_yaw,a.layout.tray_yaw)

    def test_unreachable_and_invalid_targets(self):
        ik=SO101IK(ASSETS)
        self.assertFalse(ik.solve([0.,2.,.1]).success)
        with self.assertRaises(ValueError):ik.solve([0.,float('nan'),0.])

    def test_float_camera_deadline_regression(self):
        # Match the float division used by ChOptixEngine, including deadlines
        # that round above Python's exact control time (the original 8.1 s bug).
        class Buffer:
            def HasData(self):return True
            def GetRGBA8Data(self):return np.zeros((2,2,4),dtype=np.uint8)
        class Camera:
            def __init__(self,count):self.count=count;self.data=Buffer();self.data.TimeStamp=-1.
            def GetNumLaunches(self):return self.count
            def GetUpdateRate(self):return 10.
            def GetMostRecentRGBA8Buffer(self):return self.data
        class System:
            def SetChTime(self,t):self.t=t
        for count in [81,1001,100001]:
            system=System();cameras={'wrist':Camera(count),'top_down':Camera(count)}
            class Manager:
                def Update(self):
                    for camera in cameras.values():
                        deadline=float(np.float32(camera.count)/np.float32(10.))
                        if system.t>deadline-1e-7:
                            camera.count+=1
                            camera.data.TimeStamp=float(np.float32(system.t))
                            camera.data.LaunchedCount=camera.count
            t=count/10
            images,stamps,launches=capture_synchronized(system,Manager(),cameras,t)
            self.assertEqual(system.t,t)
            self.assertEqual(launches,dict(wrist=count+1,top_down=count+1))
            self.assertEqual(stamps['wrist'],stamps['top_down'])
            self.assertEqual(images['wrist'].shape,(2,2,3))

if __name__=='__main__':unittest.main()
