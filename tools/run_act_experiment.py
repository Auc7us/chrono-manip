#!/usr/bin/env python3
"""Finish local ACT preparation, training, held-out scoring and physical rollouts."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--wait-for-exports',type=float,default=0)
    p.add_argument('--steps',type=int,default=20000);p.add_argument('--rollouts-per-set',type=int,default=5);a=p.parse_args()
    os.chdir(ROOT);setup=ROOT/'output/act_setup';setup.mkdir(parents=True,exist_ok=True)
    run=ROOT/'output/act_runs/raised_act_001';state=setup/'experiment_status.json'
    env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='2')
    def status(stage,**values):
        state.write_text(json.dumps(dict(stage=stage,time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**values),indent=2)+'\n')
        print(stage,flush=True)
    def execute(stage,command):
        status(stage,command=command)
        with (setup/(stage+'.log')).open('w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env=env,check=True)
    try:
        if not (ROOT/'output/act_data/train/chrono_metadata.json').exists():
            shards=[ROOT/f'output/act_data/train_shard_{i}' for i in range(6)]
            status('waiting_for_exports');deadline=time.monotonic()+a.wait_for_exports
            while not all((s/'chrono_metadata.json').is_file() for s in shards):
                if time.monotonic()>deadline:raise TimeoutError('Training export shards are not complete')
                time.sleep(5)
            execute('merge_training_exports',[sys.executable,'tools/merge_act_exports.py','output/act_data/train',*[str(s) for s in shards]])
        execute('train_full',['bash','train-act-baseline.sh',f'--steps={a.steps}'])
        checkpoint=run/'checkpoints/last/pretrained_model'
        execute('evaluate_held_out',[sys.executable,'tools/evaluate_act_offline.py','--checkpoint',str(checkpoint),
            '--dataset','output/act_data/validation','--output',str(run/'validation.json')])
        for name,extra in [('held_out',['--held-out-split','output/act_data/split/split.json']),('fresh',[])]:
            execute('rollout_'+name,[sys.executable,'tools/rollout_act.py','--checkpoint',str(checkpoint),
                '--output',str(run/('rollouts_'+name)),'--episodes',str(a.rollouts_per_set),'--record-episodes','3',*extra])
        execute('comparison_videos',[sys.executable,'tools/compare_act_rollouts.py',str(run/'rollouts_held_out')])
        status('complete',run=str(run),checkpoint=str(checkpoint))
    except BaseException as error:
        status('failed',error=repr(error));raise


if __name__=='__main__':main()
