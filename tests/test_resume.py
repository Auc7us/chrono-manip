"""Resume keeps completed data, seeded layout history and original batch options."""
import contextlib
import fcntl
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
from collection_resume import read_resume_state,prepare_recording_resume,checkpoint_for
from pick_place import PickPlaceController,EpisodePlan

ROOT=Path(__file__).resolve().parents[1]


def recording(root,complete=1,partial=True):
    entries=[]
    for index in range(complete+int(partial)):
        directory=root/'episodes'/f'{index:06d}';directory.mkdir(parents=True)
        info=dict(episode_index=index,complete=index<complete,success=False,frames=1)
        (directory/'episode.json').write_text(json.dumps(info));np.savez_compressed(directory/'trajectory.npz',timestamp=[0.])
        entries.append(dict(index=index,path=str(directory.relative_to(root)),frames=1,complete=info['complete'],success=False))
    data=dict(format='attempt-codex-episodes-v1',fps=30,seed=42,cameras=['wrist','right_45'],episodes=entries)
    (root/'dataset.json').write_text(json.dumps(data));return data


class HistoryController(PickPlaceController):
    """Use the actual seeded IK planner without creating a Chrono world."""
    def start_episode(self,t,layout=None):
        self.episode_index+=1;self.start_time=t
        self.plan=self.make_plan(self.episode_index) if layout is None else EpisodePlan(self.scene.assets,layout,start_pose=self.args.start_pose,tray_style=self.args.tray_style)
        self.tray_xy=self.plan.layout.tray_xy.copy();self.tray_yaw=self.plan.layout.tray_yaw


class ResumeTests(unittest.TestCase):
    def test_partial_attempts_archived_and_completed_bytes_untouched(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);recording(root)
            original={p:p.read_bytes() for p in (root/'episodes/000000').iterdir()}
            orphan=root/'episodes/000002';orphan.mkdir();(orphan/'partial.png').write_bytes(b'partial')
            state=read_resume_state(root);self.assertEqual(state['next_index'],1)
            manifest,archive=prepare_recording_resume(state)
            self.assertEqual(len(manifest['episodes']),1)
            self.assertTrue((archive/'000001/trajectory.npz').exists())
            self.assertEqual((archive/'000002/partial.png').read_bytes(),b'partial')
            self.assertFalse((root/'episodes/000001').exists())
            for path,data in original.items():self.assertEqual(path.read_bytes(),data)

    def test_completed_episode_written_before_manifest_update_is_recovered(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=recording(root,complete=2,partial=False)
            data['episodes']=data['episodes'][:1];(root/'dataset.json').write_text(json.dumps(data))
            state=read_resume_state(root);self.assertEqual(state['next_index'],2)
            self.assertEqual([e['index'] for e in state['completed_entries']],[0,1])

    def test_missing_completed_data_is_not_replaced(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);recording(root,partial=False);(root/'episodes/000000/trajectory.npz').unlink()
            before=(root/'dataset.json').read_bytes()
            with self.assertRaisesRegex(ValueError,'missing trajectory'):read_resume_state(root)
            self.assertEqual((root/'dataset.json').read_bytes(),before)

    def test_legacy_and_checkpoint_resume_preserve_future_rng_and_tray_schedule(self):
        args=SimpleNamespace(seed=42,tray_every=2,replay_episode=None,start_pose='folded',fixed_cube=False,tray_style='raised')
        scene=SimpleNamespace(assets=json.loads((ROOT/'assets/scene_assets.json').read_text()))
        base=HistoryController(scene,args);infos=[];layouts=[];checkpoints=[]
        for index in range(7):
            if index:base.start_episode(0.)
            layout=base.plan.layout
            layouts.append(np.r_[layout.cube_xy,layout.cube_yaw,layout.tray_xy,layout.tray_yaw])
            infos.append(dict(cube_spawn_xy_m=layout.cube_xy.tolist(),cube_spawn_yaw_rad=layout.cube_yaw,
                tray_xy_m=layout.tray_xy.tolist(),tray_yaw_rad=layout.tray_yaw))
            checkpoints.append(checkpoint_for(base))
        for checkpoint in [None,checkpoints[2],checkpoints[3]]:
            with contextlib.redirect_stdout(io.StringIO()):
                resumed=HistoryController(scene,args,resume=dict(next_index=3,completed_infos=infos[:3],checkpoint=checkpoint))
            for index in range(3,7):
                if index>3:resumed.start_episode(0.)
                layout=resumed.plan.layout;actual=np.r_[layout.cube_xy,layout.cube_yaw,layout.tray_xy,layout.tray_yaw]
                np.testing.assert_array_equal(actual,layouts[index]);self.assertEqual(resumed.episode_index,index)

    def batch_fixture(self,root,pid=None):
        recording(root/'worker_000',partial=False)
        batch=dict(format='attempt-codex-parallel-v1',status='interrupted',requested_episodes=1,threads_per_worker=1,
            shared_demo_args=['--rate','30','--tray-style','raised'],elapsed_s=10.,
            workers=[dict(id=0,seed=42,requested_episodes=1,directory='worker_000',log='worker_000.log',pid=pid,status='interrupted')])
        (root/'batch.json').write_text(json.dumps(batch));return batch

    def test_resume_already_finished_batch_does_not_start_workers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.batch_fixture(root);before=(root/'worker_000/episodes/000000/trajectory.npz').read_bytes()
            result=subprocess.run([sys.executable,'tools/collect_parallel.py','--resume','--output',str(root)],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('already complete',result.stdout)
            self.assertEqual(json.loads((root/'batch.json').read_text())['completed_episodes'],1)
            self.assertEqual((root/'worker_000/episodes/000000/trajectory.npz').read_bytes(),before)

    def test_resume_rejects_option_changes_before_modifying_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.batch_fixture(root);before=(root/'batch.json').read_bytes()
            result=subprocess.run([sys.executable,'tools/collect_parallel.py','--resume','--output',str(root),'--episodes','500'],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0);self.assertIn('Cannot change --episodes',result.stderr)
            self.assertEqual((root/'batch.json').read_bytes(),before)

    def test_coordinator_lock_rejects_concurrent_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.batch_fixture(root);before=(root/'batch.json').read_bytes()
            with (root/'.collector.lock').open('a+') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                result=subprocess.run([sys.executable,'tools/collect_parallel.py','--resume','--output',str(root)],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0);self.assertIn('already using this batch',result.stderr)
            self.assertEqual((root/'batch.json').read_bytes(),before)


if __name__=='__main__':unittest.main()
