"""Collect independent headless episodes in separate processes; no CUDA imports here."""
import argparse
import fcntl
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from collection_resume import read_resume_state,worker_is_running
MANAGED={'--episodes','--seed','--collect-dir','--save-dir','--record-dir','--duration',
         '--no-sensors','--validate','--motion','--assembly-pose','--auto','--headless','--no-realtime','--resume'}

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

def run_batch(args,root,extras,previous=None):
    environment=os.environ.copy();environment['PYTHONUNBUFFERED']='1'
    for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS',
                 'VECLIB_MAXIMUM_THREADS','ATTEMPT_CODEX_ENCODER_THREADS']:
        environment[name]=str(args.threads_per_worker)
    workers=[] if previous is None else [dict(w) for w in previous['workers']]
    processes={};logs=[];count=min(args.workers,args.episodes)
    now=datetime.now(timezone.utc).isoformat();previous_elapsed=0. if previous is None else previous.get('elapsed_s',0.)
    batch={} if previous is None else dict(previous)
    if previous:
        batch.setdefault('resume_sessions',[]).append(dict(resumed_utc=now,previous_status=previous['status']))
    batch.update(format='attempt-codex-parallel-v1',status='running',workers=workers,
        requested_workers=args.workers,requested_episodes=args.episodes,threads_per_worker=args.threads_per_worker,
        shared_demo_args=extras,started_utc=batch.get('started_utc',now))
    batch.pop('merge_error',None)
    write_json(root/'batch.json',batch)
    start=time.monotonic();interrupted=False
    try:
        for i in range(count):
            episodes=args.episodes//count+(i<args.episodes%count)
            directory=f'worker_{i:03d}'
            resume_worker=bool(previous and (root/directory/'dataset.json').is_file())
            completed=read_resume_state(root/directory)['next_index'] if resume_worker else 0
            if previous:
                worker=workers[i]
                if worker['directory']!=directory or worker['seed']!=args.seed+i or worker['requested_episodes']!=episodes:
                    raise ValueError('Saved worker allocation differs; refusing to change the batch')
                if completed>=episodes:
                    worker.update(status='complete',returncode=0)
                    print(f'Worker {i}: already complete ({completed}/{episodes})',flush=True);continue
            else:
                worker=dict(id=i,seed=args.seed+i,requested_episodes=episodes,directory=directory,log=f'{directory}.log',status='pending',pid=None)
                workers.append(worker)
            command=[sys.executable,str(ROOT/'demo_attempt_codex.py'),'--auto','--headless','--no-realtime',
                '--episodes',str(episodes),'--seed',str(args.seed+i),'--collect-dir',str(root/directory),*extras]
            if resume_worker:command.append('--resume')
            log=(root/f'{directory}.log').open('a' if previous else 'w');logs.append(log)
            process=subprocess.Popen(command,cwd=ROOT,env=environment,stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            processes[i]=process
            worker.update(pid=process.pid,status='running',started_elapsed_s=time.monotonic()-start)
            print(f'Worker {i}: {episodes-completed} episodes remaining, seed {args.seed+i}, log {directory}.log',flush=True)
            write_json(root/'batch.json',batch)
        pending=set(processes)
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
        stop_workers(list(processes.values()))
        for log in logs:log.close()
        for worker in workers:
            process=processes.get(worker['id'])
            if process is None:continue
            if worker['status']=='running':worker.update(status='interrupted',returncode=process.returncode)
        batch['last_session_elapsed_s']=time.monotonic()-start
        batch['elapsed_s']=previous_elapsed+batch['last_session_elapsed_s']
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


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,help='default 4; restored from batch on resume')
    parser.add_argument('--episodes',type=int,help='total attempt target, default 100; restored on resume')
    parser.add_argument('--seed',type=int,help='worker i uses seed + i; default 42')
    parser.add_argument('--threads-per-worker',type=int,help='default 1; restored on resume')
    parser.add_argument('--output',type=Path,required=True,help='batch directory; existing data requires --resume')
    parser.add_argument('--resume',action='store_true',help='continue the saved batch with its original target, workers, seeds and scene settings')
    parser.add_argument('demo_args',nargs=argparse.REMAINDER)
    args=parser.parse_args();root=args.output.resolve();previous=None
    extras=args.demo_args[1:] if args.demo_args[:1]==['--'] else args.demo_args
    for argument in extras:
        if argument.split('=',1)[0] in MANAGED:parser.error(f'{argument} is controlled by the batch launcher')
    if args.resume:
        previous=json.loads((root/'batch.json').read_text())
        if previous.get('format')!='attempt-codex-parallel-v1':parser.error('Unsupported batch format')
        if any(worker_is_running(w,root/w['directory']) for w in previous['workers']):parser.error('Batch workers are still running. Stop with Ctrl-C and wait for the launcher to exit before resuming.')
        saved=dict(workers=previous.get('requested_workers',len(previous['workers'])),episodes=previous['requested_episodes'],
            seed=previous['workers'][0]['seed'],threads_per_worker=previous['threads_per_worker'])
        for name,value in saved.items():
            requested=getattr(args,name)
            if requested is not None and requested!=value:parser.error(f'Cannot change --{name.replace("_","-")} on resume; omit it to restore the saved value')
            setattr(args,name,value)
        if extras and extras!=previous['shared_demo_args']:parser.error('Cannot change demo settings on resume; omit arguments after -- to restore the saved settings')
        extras=previous['shared_demo_args']
        for worker in previous['workers']:
            directory=root/worker['directory']
            if not directory.resolve().is_relative_to(root):parser.error('Invalid saved worker directory')
            if directory.exists():read_resume_state(directory)
    else:
        for name,value in dict(workers=4,episodes=100,seed=42,threads_per_worker=1).items():
            if getattr(args,name) is None:setattr(args,name,value)
    if args.workers<1 or args.episodes<1 or args.threads_per_worker<1 or args.seed<0:
        parser.error('workers, episodes and threads-per-worker must be positive; seed must be nonnegative')
    if not args.resume:root.mkdir(parents=True,exist_ok=False)
    with (root/'.collector.lock').open('a+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:parser.error('Another launcher is already using this batch')
        return run_batch(args,root,extras,previous)

if __name__=='__main__':sys.exit(main())
