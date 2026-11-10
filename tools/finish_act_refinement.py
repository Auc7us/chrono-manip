#!/usr/bin/env python3
"""Evaluate and package the lower-learning-rate local ACT refinement."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def main():
    os.chdir(ROOT);run=ROOT/'output/act_runs/raised_act_refine_001';setup=ROOT/'output/act_setup'
    state=setup/'experiment_status.json';env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='2')
    def status(stage,**extra):
        state.write_text(json.dumps(dict(stage=stage,run=str(run),time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**extra),indent=2)+'\n')
        print(stage,flush=True)
    def execute(stage,command):
        status(stage,command=command)
        with (setup/(stage+'.log')).open('w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env=env,check=True)
    try:
        status('refinement_training',baseline_note='20k baseline trained; 5 held-out tests failed. Fresh testing stopped after 1 completed failure to prioritize refinement.')
        deadline=time.monotonic()+1800;checkpoint=run/'checkpoints/last/pretrained_model'
        while checkpoint.resolve().parent.name!='006000' or not checkpoint.is_dir():
            if time.monotonic()>deadline:raise TimeoutError('Refinement did not finish within 30 minutes')
            time.sleep(5)
        execute('package_refined_policy',[sys.executable,'tools/package_act_policy.py',str(checkpoint),str(run/'policy')])
        model=run/'policy'
        execute('evaluate_refined_held_out',[sys.executable,'tools/evaluate_act_offline.py','--checkpoint',str(model),
            '--dataset','output/act_data/validation','--output',str(run/'validation.json')])
        for name,extra in [('held_out',['--held-out-split','output/act_data/split/split.json']),('fresh',['--seed','2000'])]:
            execute('rollout_refined_'+name,[sys.executable,'tools/rollout_act.py','--checkpoint',str(model),
                '--output',str(run/('rollouts_'+name)),'--episodes','3','--record-episodes','3',*extra])
        execute('comparison_refined',[sys.executable,'tools/compare_act_rollouts.py',str(run/'rollouts_held_out')])
        execute('report_refined',[sys.executable,'tools/report_act_experiment.py',str(run),
            '--training-log','output/act_setup/train_refine.log','--starting-step','14000'])
        status('complete',model=str(model),report=str(run/'index.html'))
    except BaseException as error:
        status('failed',error=repr(error));raise


if __name__=='__main__':main()
