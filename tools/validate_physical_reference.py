#!/usr/bin/env python3
"""Run every recorded episode through contact physics and retain all outcomes."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--episodes',type=int,nargs='+',default=list(range(10)))
    p.add_argument('--metric-fit',type=Path)
    p.add_argument('--contact-compliance',type=float)
    p.add_argument('--jaw-response',type=Path)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True);env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    def run(ep):
        out=a.output/f'episode_{ep:03d}';log=a.output/f'episode_{ep:03d}.log'
        with log.open('w') as stream:
            command=[sys.executable,str(ROOT/'tools/replay_physical_reference.py'),'--no-sensors','--episode',str(ep),'--output',str(out)]
            if a.metric_fit:command.extend(['--metric-fit',str(a.metric_fit.resolve())])
            if a.contact_compliance is not None:command.extend(['--contact-compliance',str(a.contact_compliance)])
            if a.jaw_response:command.extend(['--jaw-response',str(a.jaw_response.resolve())])
            status=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,env=env)
        if status.returncode:return dict(episode=ep,error='Replay failed',log=str(log))
        data=json.loads((out/'physics_replay.json').read_text())
        keys=['episode','cube_inside_plate','final_xy_error_m','maximum_penetration_m','maximum_cube_height_m',
              'maximum_contact_force_n','first_contact_s','bilateral_contact_render_frames']
        return {k:data[k] for k in keys}
    results=[]
    with ThreadPoolExecutor(max_workers=max(1,a.workers)) as pool:
        for future in as_completed([pool.submit(run,ep) for ep in a.episodes]):
            row=future.result();results.append(row);print(json.dumps(row),flush=True)
    results.sort(key=lambda r:r['episode'])
    (a.output/'summary.json').write_text(json.dumps(results,indent=2)+'\n')


if __name__=='__main__':main()
