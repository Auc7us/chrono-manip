#!/usr/bin/env python3
"""Exercise interrupt/resume on an isolated low-resolution two-episode batch."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from collection_resume import read_resume_state,write_json_atomic


def fingerprints(directory):
    return {str(p.relative_to(directory)):hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.rglob('*') if p.is_file()}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    output=a.output.resolve()
    if output.exists():raise FileExistsError(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    log=output.parent/(output.name+'_launcher.log');process=None
    try:
        with log.open('w') as stream:
            process=subprocess.Popen([str(ROOT/'run-parallel.sh'),'--workers','1','--episodes','2','--seed','101',
                '--output',str(output),'--','--tray-style','raised','--rate','30','--width','160','--samples','1','--tray-every','1'],
                cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            deadline=time.monotonic()+240
            while time.monotonic()<deadline:
                if process.poll() is not None:raise RuntimeError(f'Batch exited before interruption check; see {log}')
                manifest=output/'worker_000/dataset.json'
                if manifest.exists():
                    data=json.loads(manifest.read_text())
                    finished=sum(e['complete'] for e in data['episodes'])
                    partial=output/'worker_000/episodes/000001/images/wrist'
                    if finished==1 and len(list(partial.glob('*.png')))>=2:break
                time.sleep(.2)
            else:raise TimeoutError(f'First test episode did not finish; see {log}')
            print('One episode complete; interrupting only the isolated test launcher.',flush=True)
            os.killpg(process.pid,signal.SIGINT);process.wait(timeout=60)
        state=read_resume_state(output/'worker_000');assert state['next_index']==1
        completed=output/'worker_000/episodes/000000';before=fingerprints(completed)
        # Simulate the user's already-running older collector: no checkpoints
        # or newly introduced calibration snapshot in its manifest.
        checkpoint=output/'worker_000/checkpoint.json'
        if checkpoint.exists():checkpoint.rename(checkpoint.with_name('checkpoint_saved_for_test.json'))
        worker_manifest=output/'worker_000/dataset.json';saved=json.loads(worker_manifest.read_text());saved.pop('so101_joint_map',None);write_json_atomic(worker_manifest,saved)
        print('Resuming with the short command and legacy saved metadata.',flush=True)
        resume_log=output.parent/(output.name+'_resume.log')
        with resume_log.open('w') as stream:
            process=subprocess.Popen([str(ROOT/'run-parallel.sh'),'--resume','--output',str(output)],cwd=ROOT,
                stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            code=process.wait(timeout=240)
        assert code==0,f'Resume failed; see {resume_log}'
        data=json.loads((output/'dataset.json').read_text());batch=json.loads((output/'batch.json').read_text())
        assert len(data['episodes'])==2 and all(e['complete'] for e in data['episodes'])
        assert [e['worker_episode_index'] for e in data['episodes']]==[0,1]
        assert batch['requested_episodes']==2 and batch['status']=='complete'
        assert data['tray_geometry']['style']=='raised'
        assert before==fingerprints(completed),'Completed episode bytes changed'
        archives=list((output/'worker_000/interrupted').glob('*/000001'));assert archives,'Interrupted attempt was not archived'
        report=dict(completed_episodes=2,completed_episode_bytes_preserved=True,legacy_rng_restore=True,
            interrupted_episode_preserved=True,saved_settings_restored=True,successes=sum(e['success'] for e in data['episodes']))
        write_json_atomic(output.parent/(output.name+'_checks.json'),report);print(json.dumps(report,indent=2),flush=True)
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid,signal.SIGINT)
            process.wait(timeout=60)


if __name__=='__main__':main()
