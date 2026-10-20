"""Collect independent headless episodes in separate processes; no CUDA imports here."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
MANAGED={'--episodes','--seed','--collect-dir','--save-dir','--record-dir','--duration',
         '--no-sensors','--validate','--motion','--assembly-pose','--auto','--headless','--no-realtime'}

def write_json(path,data):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data,indent=2)+'\n');temporary.replace(path)

def combine_manifests(root,workers):
    """Index existing shard files without copying videos or trajectory arrays."""
    combined=None;episodes=[];seeds=[]
    for worker in workers:
        path=root/worker['directory']/'dataset.json'
        if not path.exists():continue
        manifest=json.loads(path.read_text())
        if combined is None:
            combined={k:v for k,v in manifest.items() if k not in ['episodes','seed']}
        else:
            for key,value in combined.items():
                if manifest.get(key)!=value:raise ValueError(f'Incompatible worker dataset metadata: {key}')
        seeds.append(worker['seed'])
        for episode in manifest['episodes']:
            episodes.append({**episode,'index':len(episodes),
                'path':str(Path(worker['directory'])/episode['path']),
                'worker_id':worker['id'],'worker_episode_index':episode['index'],'seed':worker['seed']})
    if combined is not None:
        combined.update(seed=None,seeds=seeds,episodes=episodes,parallel_workers=len(workers))
        write_json(root/'dataset.json',combined)
    return episodes

def stop_workers(processes):
    def send(process,number):
        if process.poll() is None:
            try:os.killpg(process.pid,number)
            except ProcessLookupError:pass
    for process in processes:
        send(process,signal.SIGINT)
    deadline=time.monotonic()+20
    while any(process.poll() is None for process in processes) and time.monotonic()<deadline:time.sleep(.1)
    for process in processes:
        send(process,signal.SIGTERM)
    deadline=time.monotonic()+5
    while any(process.poll() is None for process in processes) and time.monotonic()<deadline:time.sleep(.1)
    for process in processes:
        send(process,signal.SIGKILL)
        process.wait()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--episodes',type=int,default=100,help='total episode attempts, divided across workers')
    parser.add_argument('--seed',type=int,default=42,help='worker i uses seed + i')
    parser.add_argument('--threads-per-worker',type=int,default=1,help='OpenMP/BLAS/MP4 encoder thread limit per worker')
    parser.add_argument('--output',type=Path,required=True,help='new batch directory, never overwritten')
    parser.add_argument('demo_args',nargs=argparse.REMAINDER,help='shared demo flags after --, e.g. -- --rate 30 --global-camera right-45')
    args=parser.parse_args()
    if args.workers<1 or args.episodes<1 or args.threads_per_worker<1 or args.seed<0:
        parser.error('workers, episodes and threads-per-worker must be positive; seed must be nonnegative')
    extras=args.demo_args[1:] if args.demo_args[:1]==['--'] else args.demo_args
    for argument in extras:
        if argument.split('=',1)[0] in MANAGED:parser.error(f'{argument} is controlled by the batch launcher')
    root=args.output.resolve()
    root.mkdir(parents=True,exist_ok=False)
    count=min(args.workers,args.episodes)
    environment=os.environ.copy();environment['PYTHONUNBUFFERED']='1'
    for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS',
                 'VECLIB_MAXIMUM_THREADS','ATTEMPT_CODEX_ENCODER_THREADS']:
        environment[name]=str(args.threads_per_worker)
    workers=[];processes=[];logs=[]
    batch=dict(format='attempt-codex-parallel-v1',status='running',workers=workers,
        requested_episodes=args.episodes,threads_per_worker=args.threads_per_worker,
        shared_demo_args=extras,started_utc=datetime.now(timezone.utc).isoformat())
    write_json(root/'batch.json',batch)
    start=time.monotonic();interrupted=False
    try:
        for i in range(count):
            episodes=args.episodes//count+(i<args.episodes%count)
            directory=f'worker_{i:03d}'
            command=[sys.executable,str(ROOT/'demo_attempt_codex.py'),'--auto','--headless','--no-realtime',
                '--episodes',str(episodes),'--seed',str(args.seed+i),'--collect-dir',str(root/directory),*extras]
            log=(root/f'{directory}.log').open('w');logs.append(log)
            process=subprocess.Popen(command,cwd=ROOT,env=environment,stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            processes.append(process)
            workers.append(dict(id=i,seed=args.seed+i,requested_episodes=episodes,directory=directory,
                log=f'{directory}.log',pid=process.pid,status='running',started_elapsed_s=time.monotonic()-start))
            print(f'Worker {i}: {episodes} episodes, seed {args.seed+i}, log {directory}.log',flush=True)
            write_json(root/'batch.json',batch)
        pending=set(range(count))
        while pending:
            for i in list(pending):
                code=processes[i].poll()
                if code is None:continue
                workers[i].update(status='complete' if code==0 else 'failed',returncode=code,
                                  elapsed_s=time.monotonic()-start-workers[i]['started_elapsed_s'])
                pending.remove(i);write_json(root/'batch.json',batch)
                print(f'Worker {i} {workers[i]["status"]} ({workers[i]["elapsed_s"]:.1f} s)',flush=True)
            if pending:time.sleep(.2)
    except KeyboardInterrupt:
        interrupted=True;batch['status']='interrupted'
        print('Stopping recording workers; keeping finished and partial episodes.',flush=True)
    finally:
        stop_workers(processes)
        for log in logs:log.close()
        for worker,process in zip(workers,processes):
            if worker['status']=='running':worker.update(status='interrupted',returncode=process.returncode)
        batch['elapsed_s']=time.monotonic()-start
        try:episodes=combine_manifests(root,workers)
        except (OSError,ValueError) as error:
            episodes=[];batch['merge_error']=str(error)
            print(f'Combined index failed: {error}; worker recordings remain in their folders.',file=sys.stderr,flush=True)
        completed=sum(bool(e['complete']) for e in episodes)
        successful=sum(bool(e['complete'] and e['success']) for e in episodes)
        batch.update(recorded_episodes=len(episodes),completed_episodes=completed,successful_episodes=successful,
            completed_episodes_per_hour=completed*3600/batch['elapsed_s'],
            successful_episodes_per_hour=successful*3600/batch['elapsed_s'])
        if not interrupted:
            batch['status']='complete' if not batch.get('merge_error') and len(workers)==count and all(w['status']=='complete' for w in workers) else 'failed'
        write_json(root/'batch.json',batch)
    print(f'{successful}/{completed} completed episodes succeeded in {batch["elapsed_s"]:.1f} s; '
          f'{batch["successful_episodes_per_hour"]:.1f} successful episodes/hour. Dataset: {root}',flush=True)
    if interrupted:return 130
    return 0 if batch['status']=='complete' else 1

if __name__=='__main__':sys.exit(main())
