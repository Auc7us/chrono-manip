"""Projection conventions, physical landmark fit and default camera selection."""
import unittest
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
from reference_camera import DEFAULT_PROFILE,PHOTO_FIT_PROFILE,load_profile,load_wrist_profile,project_world,pixel_rays,camera_metadata,validate_lens


class ReferenceCameraTests(unittest.TestCase):
    def test_default_and_optional_camera(self):
        from demo_attempt_codex import parse_args
        with patch('sys.argv',['demo']):args=parse_args()
        self.assertEqual(args.global_camera,'right-45')
        self.assertEqual(args.lens_model,'reference')
        self.assertEqual(args.camera_profile,DEFAULT_PROFILE)
        self.assertLess(camera_metadata(args)['estimated_pose_landmark_rms_px'],4.)
        with patch('sys.argv',['demo','--camera-profile',str(PHOTO_FIT_PROFILE)]):args=parse_args()
        self.assertEqual(args.camera_profile,PHOTO_FIT_PROFILE)
        self.assertLess(camera_metadata(args)['estimated_pose_landmark_rms_px'],3.)
        with patch('sys.argv',['demo','--global-camera','top-down']):args=parse_args()
        self.assertEqual(args.global_camera,'top-down')

    def test_observed_box_and_sheet_landmarks(self):
        for path in [DEFAULT_PROFILE,PHOTO_FIT_PROFILE]:
            profile=load_profile(path);landmarks=list(profile['landmarks'].values())
            predicted=project_world([v['world_m'] for v in landmarks],profile)
            reference=np.array([v['reference_px'] for v in landmarks])
            errors=np.linalg.norm(predicted-reference,axis=1)
            self.assertLess(np.sqrt(np.mean(errors**2)),4. if path==DEFAULT_PROFILE else 3.)
            self.assertLess(errors.max(),6.)

    def test_pixel_rays_have_correct_origin_and_orientation(self):
        profile=load_profile();center=profile['lens']['principal_point_px']
        np.testing.assert_allclose(pixel_rays([center])[0],np.array(profile['world_from_optical_rotation'])[:,0],atol=1e-12)
        pixels=np.array([[x,y] for x in np.linspace(0,640,21) for y in np.linspace(0,480,17)])
        rays=pixel_rays(pixels);np.testing.assert_allclose(np.linalg.norm(rays,axis=1),1,atol=1e-12)
        world=np.array(profile['optical_position_m'])+rays*.6
        np.testing.assert_allclose(project_world(world),pixels,atol=1e-8)

    def test_noninvertible_corner_projection_is_rejected(self):
        invalid=dict(focal_px=275.,principal_point_px=[329.,242.],theta_coefficients=[.05,-.038])
        with self.assertRaises(ValueError):validate_lens(invalid)

    def test_independent_wrist_optics_and_pose(self):
        profile=load_wrist_profile();frame=np.array(profile['link_from_optical'])
        np.testing.assert_allclose(frame[:3,:3].T@frame[:3,:3],np.eye(3),atol=1e-10)
        # The fit now also constrains CAD clearance against the physical floor;
        # allow the measured subpixel tradeoff relative to the image-only fit.
        self.assertLess(profile['validation']['held_out']['mat_boundary_rms_px'],5.)
        self.assertLess(profile['validation']['held_out']['fixed_jaw_rms_px'],1.5)
        profile={**profile,'optical_position_m':frame[:3,3],'world_from_optical_rotation':frame[:3,:3]}
        pixels=np.array([[5,5],[635,475],[320,240],[150,350]],dtype=float)
        rays=pixel_rays(pixels,profile);points=frame[:3,3]+rays*.4
        np.testing.assert_allclose(project_world(points,profile),pixels,atol=1e-8)
        from demo_attempt_codex import parse_args
        args=parse_args(['--width','1280'])
        metadata=camera_metadata(args,'wrist')
        self.assertAlmostEqual(metadata['output_focal_px'],profile['lens']['focal_px']*2)
        self.assertNotEqual(metadata['fitted_lens'],camera_metadata(args)['fitted_lens'])

    def test_motor_scales_remain_consistent_with_recorded_encoder_grid(self):
        root=Path(__file__).resolve().parents[1]
        units=json.loads((root/'assets/encoder_units.json').read_text())
        mapping=json.loads((root/'assets/video_joint_map.json').read_text())
        np.testing.assert_allclose(mapping['scale'],units['signed_scale'],atol=1e-6)
        for motor in units['motors']:
            self.assertLess(motor['grid_99th_percentile_error_ticks'],.001)

    def test_recorded_jaws_avoid_large_solid_floor_intersections(self):
        from tools.audit_jaw_clearance import audit
        root=Path(__file__).resolve().parents[1];trajectory=root/'output/video_fit/reference_trajectory.npz'
        if not trajectory.exists():self.skipTest('Prepared reference trajectory is not present')
        assets=json.loads((root/'assets/scene_assets.json').read_text())
        mapping=json.loads((root/'assets/video_joint_map.json').read_text())
        with np.load(trajectory) as data:report=audit(assets,mapping,data)
        # The 1 mm mat can compress; allow 0.5 mm of CAD/floor uncertainty.
        # The old image-only mapping put the solid jaws nearly 10 mm below Z=0.
        for episode in report['episodes']:
            for name in ['fixed_jaw','moving_jaw']:
                self.assertGreater(episode[name]['minimum_world_z_m'],-.0005,(episode['episode'],name))


if __name__=='__main__':unittest.main()
