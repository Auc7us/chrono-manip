"""Dataset shard integrity and process cleanup checks; no Chrono/GPU needed."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from tools.collect_parallel import combine_manifests,stop_workers

class ParallelCollectionTests(unittest.TestCase):
    def test_index_shards_without_colliding_local_episode_numbers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);workers=[];originals=[]
            for i in range(2):
                directory=f'worker_{i:03d}';path=root/directory/'dataset.json';path.parent.mkdir()
                data=dict(format='attempt-codex-episodes-v1',fps=30,seed=42+i,cameras=['wrist','right_45'],
                    episodes=[dict(index=0,path='episodes/000000',frames=300,success=i==0,complete=True)])
                path.write_text(json.dumps(data));originals.append((path,path.read_bytes()))
                workers.append(dict(id=i,seed=42+i,directory=directory))
            episodes=combine_manifests(root,workers)
            self.assertEqual([e['index'] for e in episodes],[0,1])
            self.assertEqual([e['worker_episode_index'] for e in episodes],[0,0])
            self.assertEqual([e['path'] for e in episodes],['worker_000/episodes/000000','worker_001/episodes/000000'])
            result=json.loads((root/'dataset.json').read_text())
            self.assertEqual(result['seeds'],[42,43])
            self.assertEqual([e['success'] for e in result['episodes']],[True,False])
            for path,content in originals:self.assertEqual(path.read_bytes(),content)

    def test_mismatched_camera_metadata_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);workers=[]
            for i,cameras in enumerate([['wrist','top_down'],['wrist','right_45']]):
                directory=f'worker_{i:03d}';path=root/directory/'dataset.json';path.parent.mkdir()
                path.write_text(json.dumps(dict(fps=30,cameras=cameras,seed=i,episodes=[])))
                workers.append(dict(id=i,seed=i,directory=directory))
            with self.assertRaisesRegex(ValueError,'cameras'):combine_manifests(root,workers)
            self.assertFalse((root/'dataset.json').exists())

    def test_managed_output_flag_rejected_before_starting_workers(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'new'
            result=subprocess.run([sys.executable,'tools/collect_parallel.py','--output',str(output),
                '--','--collect-dir','shared'],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('controlled by the batch launcher',result.stderr)
            self.assertFalse(output.exists())

    def test_stop_recording_child_process_group(self):
        process=subprocess.Popen([sys.executable,'-c','import signal; signal.pause()'],
            start_new_session=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        try:
            stop_workers([process])
            self.assertIsNotNone(process.returncode)
        finally:
            if process.poll() is None:process.kill();process.wait()

    def test_existing_batch_directory_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'existing';output.mkdir()
            sentinel=output/'keep.txt';sentinel.write_text('original')
            result=subprocess.run([sys.executable,'tools/collect_parallel.py','--workers','1',
                '--episodes','1','--output',str(output)],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertEqual(sentinel.read_text(),'original')
            self.assertEqual(list(output.iterdir()),[sentinel])

if __name__=='__main__':unittest.main()
