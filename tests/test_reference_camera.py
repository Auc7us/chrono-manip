"""Projection conventions, physical landmark fit and default camera selection."""
import unittest
from unittest.mock import patch
import numpy as np
from reference_camera import DEFAULT_PROFILE,PHOTO_FIT_PROFILE,load_profile,project_world,pixel_rays,camera_metadata


class ReferenceCameraTests(unittest.TestCase):
    def test_default_and_optional_camera(self):
        from demo_attempt_codex import parse_args
        with patch('sys.argv',['demo']):args=parse_args()
        self.assertEqual(args.global_camera,'right-45')
        self.assertEqual(args.lens_model,'reference')
        self.assertEqual(args.camera_profile,DEFAULT_PROFILE)
        self.assertIsNone(camera_metadata(args)['estimated_pose_landmark_rms_px'])
        with patch('sys.argv',['demo','--camera-profile',str(PHOTO_FIT_PROFILE)]):args=parse_args()
        self.assertEqual(args.camera_profile,PHOTO_FIT_PROFILE)
        self.assertLess(camera_metadata(args)['estimated_pose_landmark_rms_px'],3.)
        with patch('sys.argv',['demo','--global-camera','top-down']):args=parse_args()
        self.assertEqual(args.global_camera,'top-down')

    def test_observed_box_and_sheet_landmarks(self):
        profile=load_profile(PHOTO_FIT_PROFILE);landmarks=list(profile['landmarks'].values())
        predicted=project_world([v['world_m'] for v in landmarks],profile)
        reference=np.array([v['reference_px'] for v in landmarks])
        errors=np.linalg.norm(predicted-reference,axis=1)
        self.assertLess(np.sqrt(np.mean(errors**2)),3.)
        self.assertLess(errors.max(),5.)

    def test_pixel_rays_have_correct_origin_and_orientation(self):
        profile=load_profile();center=profile['lens']['principal_point_px']
        np.testing.assert_allclose(pixel_rays([center])[0],np.array(profile['world_from_optical_rotation'])[:,0],atol=1e-12)
        pixels=np.array([[1,1],[639,479],[320,240],[100,400],[500,25]],dtype=float)
        rays=pixel_rays(pixels);np.testing.assert_allclose(np.linalg.norm(rays,axis=1),1,atol=1e-12)
        world=np.array(profile['optical_position_m'])+rays*.6
        np.testing.assert_allclose(project_world(world),pixels,atol=1e-8)


if __name__=='__main__':unittest.main()
