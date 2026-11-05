"""Reference-coordinate round trips and state/action export schema checks."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
from motor_calibration import NAMES
from so101_format import joint_map_snapshot,so101_conversion,convert_joints,check_gripper_range,SO101_UNITS

ROOT=Path(__file__).resolve().parents[1]


def manifest():
    return dict(joint_units='radians_offset_from_STEP_pose',joint_names=list(NAMES),cameras=['wrist','right_45'])


class SO101FormatTests(unittest.TestCase):
    def test_reference_recordings_round_trip_including_gripper(self):
        try:import pyarrow.parquet as pq
        except ImportError:self.skipTest('PyArrow is not installed in this test environment')
        files=sorted((ROOT.parent/'ref_data/green_cube_red_plate_1/data').rglob('*.parquet'))
        if not files:self.skipTest('Original reference Parquet files are not present')
        conversion=so101_conversion(manifest());mapping=conversion['cad_joint_map'];count=0
        for file in files:
            table=pq.read_table(file,columns=['observation.state','action'])
            for column in ['observation.state','action']:
                real=np.array(table[column].to_pylist())
                native=np.deg2rad(real*mapping['scale']+mapping['offset_deg'])
                np.testing.assert_allclose(convert_joints(native,conversion),real,atol=1e-10)
            count+=len(table)
        self.assertEqual(count,6632)

    def test_recorded_map_is_independent_of_later_asset_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'map.json';path.write_text(json.dumps(dict(scale=[-1,1,1,1,-1,-2],offset_deg=[0,90,-90,0,90,100])))
            saved=manifest();saved['so101_joint_map']=joint_map_snapshot(path)
            path.write_text(json.dumps(dict(scale=[1]*6,offset_deg=[0]*6)))
            conversion=so101_conversion(saved)
            native=np.deg2rad([0,90,-90,0,90,50])
            np.testing.assert_allclose(convert_joints(native,conversion),[0,0,0,0,0,25],atol=1e-12)
            self.assertEqual(conversion['calibration_origin'],'recorded_calibration_snapshot')

    def test_joint_reordering_and_velocity_without_position_offset(self):
        saved=manifest();saved['joint_names']=list(reversed(NAMES));conversion=so101_conversion(saved)
        mapping=conversion['cad_joint_map'];real=np.array([10,20,30,40,50,60.])
        native=np.deg2rad(real*mapping['scale']+mapping['offset_deg'])
        np.testing.assert_allclose(convert_joints(native[::-1],conversion),real,atol=1e-10)
        native_velocity=np.deg2rad(np.array(mapping['scale']))
        np.testing.assert_allclose(convert_joints(native_velocity[::-1],conversion,velocity=True),np.ones(6),atol=1e-12)

    def test_reject_double_conversion_overhead_alias_and_invalid_gripper(self):
        saved=manifest();saved['joint_units']=SO101_UNITS
        with self.assertRaisesRegex(ValueError,'already normalized'):so101_conversion(saved)
        saved=manifest();saved['cameras']=['wrist','top_down']
        with self.assertRaisesRegex(ValueError,'overhead'):so101_conversion(saved)
        with self.assertRaisesRegex(ValueError,'not silently clipped'):check_gripper_range([[0,0,0,0,0,101]])

    def test_export_writer_receives_reference_schema_and_distinct_absolute_targets(self):
        from tools.export_lerobot import main
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'native';episode=source/'episodes/000000';episode.mkdir(parents=True)
            output=Path(directory)/'lerobot';saved=manifest();saved['so101_joint_map']=joint_map_snapshot()
            saved.update(fps=30,grasp_mode='physical_jaw_contacts',action_semantics='absolute_joint_target_for_next_control_tick',
                episodes=[dict(path='episodes/000000',frames=3,success=True,complete=True)])
            (source/'dataset.json').write_text(json.dumps(saved))
            (episode/'episode.json').write_text(json.dumps(dict(task='Place the cube in the red tray',success=True)))
            real=np.array([[10,20,-30,40,50,15],[11,21,-31,41,51,14],[12,22,-32,42,52,13.]])
            action=real+np.array([1,2,3,4,5,-5]);mapping=saved['so101_joint_map']
            native=lambda v:np.deg2rad(v*mapping['scale']+mapping['offset_deg'])
            np.savez_compressed(episode/'trajectory.npz',timestamp=np.arange(3)/30,
                observation_state=native(real),action=native(action),image_timestamps=np.tile(np.arange(3)[:,None]/30,(1,2)),image_launches=np.tile(np.arange(3)[:,None],(1,2)))
            for camera in saved['cameras']:
                target=episode/'images'/camera;target.mkdir(parents=True)
                for i in range(3):Image.new('RGB',(8,6),(i,0,0)).save(target/f'{i:06d}.png')
            frames=[];created={};status={}
            class Writer:
                @classmethod
                def create(cls,**kwargs):
                    created.update(kwargs);kwargs['root'].mkdir();return cls()
                def add_frame(self,frame):frames.append(frame)
                def save_episode(self):status['saved']=True
                def finalize(self):status['finalized']=True
            fake=types.ModuleType('lerobot.datasets.lerobot_dataset');fake.LeRobotDataset=Writer
            modules={'lerobot':types.ModuleType('lerobot'),'lerobot.datasets':types.ModuleType('lerobot.datasets'),'lerobot.datasets.lerobot_dataset':fake}
            with patch.dict(sys.modules,modules),patch.object(sys,'argv',['export',str(source),str(output),'--so101']),contextlib.redirect_stdout(io.StringIO()):main()
            self.assertEqual(created['robot_type'],'so_follower');self.assertEqual(created['vcodec'],'h264')
            self.assertEqual(set(created['features']),{'observation.state','action','observation.images.wrist','observation.images.external'})
            self.assertEqual(created['features']['action']['names'],[n+'.pos' for n in NAMES])
            np.testing.assert_allclose([row['observation.state'] for row in frames],real,atol=1e-6)
            np.testing.assert_allclose([row['action'] for row in frames],action,atol=1e-6)
            self.assertEqual(frames[0]['action'].dtype,np.float32)
            self.assertEqual(status,dict(saved=True,finalized=True))
            metadata=json.loads((output/'chrono_metadata.json').read_text())
            self.assertEqual(metadata['joint_units'],SO101_UNITS)
            self.assertEqual(metadata['so101_conversion']['calibration_origin'],'recorded_calibration_snapshot')
            # Coordinate-only export remains available in the Chrono container
            # without the LeRobot/PyTorch stack or copied image files.
            for image in episode.glob('images/*/*.png'):image.unlink()
            arrays_output=Path(directory)/'arrays'
            with patch.object(sys,'argv',['export',str(source),str(arrays_output),'--so101','--trajectories-only']),contextlib.redirect_stdout(io.StringIO()):main()
            with np.load(arrays_output/'episodes/000000/trajectory.npz') as converted:
                np.testing.assert_allclose(converted['observation.state'],real,atol=1e-6)
                np.testing.assert_allclose(converted['action'],action,atol=1e-6)
                np.testing.assert_array_equal(converted['timestamp'],np.arange(3)/30)
                self.assertNotIn('observation_state',converted.files)
            with np.load(episode/'trajectory.npz') as original:
                np.testing.assert_array_equal(original['observation_state'],native(real))
            self.assertFalse(json.loads((arrays_output/'conversion.json').read_text())['full_lerobot_dataset'])


if __name__=='__main__':unittest.main()
