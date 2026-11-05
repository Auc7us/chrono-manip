"""Check saved hardware normalization and leader-to-follower command transfer."""
import copy
import json
from pathlib import Path
import unittest
import numpy as np
from motor_calibration import to_ticks,from_ticks,quantize_commands,nominal_home_offsets

ROOT=Path(__file__).resolve().parents[1]


class MotorCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.profile=json.loads((ROOT/'assets/recorded_motor_calibration.json').read_text())
        self.cal=self.profile['follower']['calibration']

    def test_follower_home_uses_each_range_midpoint(self):
        home=from_ticks(np.full(6,2048.),self.cal)
        np.testing.assert_allclose(home,[7.032967032967033,4.,9.934065934065934,-3.5604395604395602,.04395604395604396,.9881422924901185])
        self.assertAlmostEqual(nominal_home_offsets(self.cal)[2],-99.93406593406593)

    def test_homing_offset_is_not_applied_twice(self):
        changed=copy.deepcopy(self.cal)
        for motor in changed.values():motor['homing_offset']+=100
        np.testing.assert_array_equal(to_ticks(np.zeros((2,6)),self.cal),to_ticks(np.zeros((2,6)),changed))

    def test_leader_gripper_target_is_sent_through_follower_range(self):
        values=np.zeros((3,6));values[:,5]=[-1,50,101]
        ticks=to_ticks(values,self.cal,command=True)
        np.testing.assert_array_equal(ticks[:,5],[2033,2792,3551])
        self.assertEqual(to_ticks(np.array([0,0,0,0,0,100]),self.profile['leader']['calibration'],command=True)[5],3255)
        np.testing.assert_allclose(quantize_commands(values,self.profile)[:,5],[0,50,100])

    def test_supplied_calibration_explains_recorded_encoder_grid(self):
        path=ROOT/'output/video_fit/reference_trajectory.npz'
        if not path.exists():self.skipTest('Prepared reference trajectory is not present')
        with np.load(path) as data:
            for role,column in [('follower','observation_state'),('leader','action')]:
                ticks=to_ticks(data[column],self.profile[role]['calibration'])
                self.assertLess(abs(ticks-np.rint(ticks)).max(),.001)
            error=abs((quantize_commands(data['action'],self.profile,recover_leader_grid=True)-data['action'])*self.profile['signed_scale'])
            self.assertLess(error.max(),360/4095+1e-6)


if __name__=='__main__':unittest.main()
