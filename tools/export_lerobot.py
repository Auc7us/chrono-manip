"""Convert native synchronized episodes with the official LeRobot writer.

This tool never uploads data. Run in a separate environment with LeRobot,
NumPy and Pillow installed. Only successful complete episodes export by default.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path);parser.add_argument('output',type=Path)
    parser.add_argument('--repo-id',default='local/attempt-codex-pick-place')
    parser.add_argument('--include-failed',action='store_true')
    parser.add_argument('--allow-assisted',action='store_true',help='explicitly allow legacy datasets that used cube attachment and no physical jaw contacts')
    parser.add_argument('--joint-map',type=Path,help='JSON with scale[6], offset[6], units; output = scale * sim_radians + offset')
    parser.add_argument('--validate-only',action='store_true')
    args=parser.parse_args()
    manifest=json.loads((args.source/'dataset.json').read_text())
    if manifest.get('grasp_mode')!='physical_jaw_contacts' and not args.allow_assisted:
        raise ValueError('Legacy assisted-grasp dataset: jaw contacts were not simulated. Recollect for physical-grasp training, '
                         'or explicitly use --allow-assisted for legacy pipeline experiments.')
    fps=float(manifest['fps'])
    if abs(fps-round(fps))>1e-9:raise ValueError('LeRobot export requires an integer FPS')
    episodes=[e for e in manifest['episodes'] if e['frames'] and (args.include_failed or (e['success'] and e['complete']))]
    if not episodes:raise ValueError('No eligible episodes; collect successful complete episodes first')
    scale=np.ones(6);offset=np.zeros(6);units=manifest['joint_units']
    if args.joint_map:
        mapping=json.loads(args.joint_map.read_text());scale=np.asarray(mapping['scale']);offset=np.asarray(mapping['offset']);units=mapping['units']
        if scale.shape!=(6,) or offset.shape!=(6,) or not np.isfinite(scale).all() or not np.isfinite(offset).all() or (scale==0).any():
            raise ValueError('Joint mapping requires six finite nonzero scales and six finite offsets')
    for episode in episodes:
        directory=args.source/episode['path'];d=np.load(directory/'trajectory.npz')
        n=len(d['timestamp'])
        if n!=episode['frames'] or not np.allclose(d['timestamp'],np.arange(n)/fps,atol=1e-6):raise ValueError('Frame count or timestamps are inconsistent')
        if d['observation_state'].shape!=(n,6) or d['action'].shape!=(n,6):raise ValueError('State/action shape must be (frames,6)')
        if not np.isfinite(d['observation_state']).all() or not np.isfinite(d['action']).all():raise ValueError('Nonfinite joint data')
        if np.max(np.ptp(d['image_timestamps'],axis=1))>1e-4:raise ValueError('Camera timestamps are not synchronized')
        if n>1 and not (np.diff(d['image_launches'],axis=0)>0).all():raise ValueError('Repeated or out-of-order images')
        for camera in manifest['cameras']:
            for i in range(n):
                if not (directory/'images'/camera/f'{i:06d}.png').is_file():raise ValueError('Missing camera image')
    first=args.source/episodes[0]['path']/'images'/manifest['cameras'][0]/'000000.png'
    width,height=Image.open(first).size
    print(f'Validated {len(episodes)} episodes, {sum(e["frames"] for e in episodes)} frames, {fps:g} FPS; joint units: {units}')
    if args.validate_only:return
    if args.output.exists():raise FileExistsError(f'Output already exists: {args.output}')
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        raise SystemExit('Install LeRobot in this Python environment to export; --validate-only does not need it.') from exc
    names=[f'{joint}.pos' for joint in manifest['joint_names']]
    features={key:dict(dtype='float32',shape=(6,),names=names) for key in ['observation.state','action']}
    features.update({f'observation.images.{camera}':dict(dtype='video',shape=(height,width,3),names=['height','width','channels']) for camera in manifest['cameras']})
    dataset=LeRobotDataset.create(repo_id=args.repo_id,root=args.output,fps=round(fps),features=features,
        robot_type='so101_pychrono',use_videos=True,image_writer_threads=2)
    metadata=[]
    try:
        for episode in episodes:
            directory=args.source/episode['path'];d=np.load(directory/'trajectory.npz');info=json.loads((directory/'episode.json').read_text())
            for i in range(len(d['timestamp'])):
                frame={'observation.state':(d['observation_state'][i]*scale+offset).astype(np.float32),
                       'action':(d['action'][i]*scale+offset).astype(np.float32),'task':info['task']}
                frame.update({f'observation.images.{camera}':np.array(Image.open(directory/'images'/camera/f'{i:06d}.png').convert('RGB')) for camera in manifest['cameras']})
                dataset.add_frame(frame)
            dataset.save_episode();metadata.append(info)
    finally:
        dataset.finalize()
    (args.output/'chrono_metadata.json').write_text(json.dumps(dict(joint_units=units,joint_scale=scale.tolist(),joint_offset=offset.tolist(),
        grasp_mode=manifest['grasp_mode'],action_semantics=manifest['action_semantics'],jaw_collision=manifest.get('jaw_collision'),
        cameras=manifest['cameras'],camera_config=manifest.get('camera_config'),
        work_surface=manifest.get('work_surface'),source_episodes=metadata),indent=2)+'\n')
    print('Exported locally to',args.output)

if __name__=='__main__':main()
