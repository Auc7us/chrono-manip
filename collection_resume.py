"""Episode-boundary resume data; completed recordings are never rewritten."""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import uuid


def write_json_atomic(path,data):
    path=Path(path);temporary=path.with_name(f'{path.name}.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(data,indent=2)+'\n');temporary.replace(path)


def read_resume_state(root):
    root=Path(root);manifest=json.loads((root/'dataset.json').read_text())
    if manifest.get('format')!='attempt-codex-episodes-v1' or manifest.get('parallel_workers'):
        raise ValueError('Expected a single-worker recording; resume a batch with run-parallel.sh')
    listed={int(e['index']):e for e in manifest['episodes']};directories={}
    for path in (root/'episodes').glob('*'):
        if path.is_dir() and path.name.isdigit():
            if not path.resolve().is_relative_to(root.resolve()):raise ValueError(f'Episode directory is outside recording: {path}')
            directories[int(path.name)]=path
    complete={}
    for index,path in directories.items():
        info_path=path/'episode.json'
        if not info_path.exists():continue
        try:info=json.loads(info_path.read_text())
        except json.JSONDecodeError:
            if listed.get(index,{}).get('complete'):raise ValueError(f'Invalid completed episode metadata: {info_path}')
            continue
        if info.get('complete'):
            if not (path/'trajectory.npz').is_file():raise ValueError(f'Completed episode is missing trajectory: {path}')
            if info.get('episode_index')!=index:raise ValueError(f'Episode index mismatch: {path}')
            complete[index]=(path,info)
    for index,entry in listed.items():
        if entry.get('complete') and index not in complete:raise ValueError(f'Completed episode files are missing: {root / entry["path"]}')
    next_index=0
    while next_index in complete:next_index+=1
    if any(i>=next_index for i in complete):raise ValueError('Completed episode history has a gap; refusing to overwrite recordings')
    entries=[];infos=[]
    for index in range(next_index):
        path,info=complete[index];infos.append(info)
        entries.append(dict(index=index,path=str(path.relative_to(root)),frames=info['frames'],success=bool(info['success']),complete=True))
    checkpoint=None
    try:checkpoint=json.loads((root/'checkpoint.json').read_text())
    except (FileNotFoundError,json.JSONDecodeError):pass
    if checkpoint and checkpoint.get('format')!='attempt-codex-resume-v1':checkpoint=None
    return dict(root=root,manifest=manifest,completed_entries=entries,completed_infos=infos,
        next_index=next_index,partial_directories=[p for i,p in sorted(directories.items()) if i>=next_index],checkpoint=checkpoint)


def prepare_recording_resume(state):
    """Archive interrupted attempts only, then index the completed prefix."""
    root=state['root'];archive=None
    if state['partial_directories']:
        name=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]
        archive=root/'interrupted'/name;archive.mkdir(parents=True)
        write_json_atomic(archive/'dataset_before_resume.json',state['manifest'])
        for path in state['partial_directories']:path.rename(archive/path.name)
    manifest={**state['manifest'],'episodes':state['completed_entries']}
    write_json_atomic(root/'dataset.json',manifest)
    return manifest,archive


def checkpoint_for(controller):
    layout=controller.plan.layout
    return dict(format='attempt-codex-resume-v1',episode_index=controller.episode_index,seed=controller.args.seed,
        rng_state=controller.rng.bit_generator.state,
        layout=dict(cube_xy=layout.cube_xy.tolist(),cube_yaw=float(layout.cube_yaw),
                    tray_xy=layout.tray_xy.tolist(),tray_yaw=float(layout.tray_yaw)))


def worker_is_running(worker,recording_root=None):
    """Also protect legacy launchers that predate the coordinator file lock."""
    pid=worker.get('pid')
    if not pid:return False
    try:command=Path(f'/proc/{int(pid)}/cmdline').read_bytes().split(b'\0')
    except (FileNotFoundError,ProcessLookupError):return False
    except PermissionError:
        try:os.kill(int(pid),0);return True
        except ProcessLookupError:return False
    if not (any(b'demo_attempt_codex.py' in value for value in command) and b'--collect-dir' in command):return False
    if recording_root is not None:
        try:
            path=Path(os.fsdecode(command[command.index(b'--collect-dir')+1]))
            if not path.is_absolute():path=Path(os.readlink(f'/proc/{int(pid)}/cwd'))/path
            return path.resolve()==Path(recording_root).resolve()
        except (OSError,IndexError):return False
    return True


def recording_is_running(root):
    root=Path(root).resolve()
    for directory in Path('/proc').glob('[0-9]*'):
        if int(directory.name)==os.getpid():continue
        try:
            command=(directory/'cmdline').read_bytes().split(b'\0')
            if not any(b'demo_attempt_codex.py' in value for value in command):continue
            position=command.index(b'--collect-dir');path=Path(os.fsdecode(command[position+1]))
            if not path.is_absolute():path=Path(os.readlink(directory/'cwd'))/path
            if path.resolve()==root:return True
        except (OSError,ValueError,IndexError):continue
    return False
