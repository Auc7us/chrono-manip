"""Convert native synchronized episodes with the official LeRobot writer.

This tool never uploads data. Run in a separate environment with LeRobot,
NumPy and Pillow installed. Only complete episodes that place the cube flat in the tray export by
default (see prepare_act_split.usable).
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from so101_format import so101_conversion,convert_joints,check_gripper_range
from tools.prepare_act_split import usable


def export_trajectories(source,output,manifest,episodes,conversion):
    """A dependency-light array translation, explicitly not a full LeRobot dataset."""
    output.mkdir(parents=True,exist_ok=False)
    metadata=dict(format='so101-trajectory-arrays-v1',full_lerobot_dataset=False,
        fps=manifest['fps'],robot_type='so_follower',joint_names=[n+'.pos' for n in conversion['joint_names']],
        joint_units=conversion['units'],per_joint_units=conversion['per_joint_units'],
        action_semantics=manifest['action_semantics'],conversion=conversion,
        source_dataset=str(source.resolve()),tray_geometry=manifest.get('tray_geometry'),episodes=[])
    for index,episode in enumerate(episodes):
        original=source/episode['path'];directory=output/'episodes'/f'{index:06d}';directory.mkdir(parents=True)
        with np.load(original/'trajectory.npz') as saved:arrays={key:saved[key] for key in saved.files}
        arrays['observation.state']=convert_joints(arrays.pop('observation_state'),conversion).astype(np.float32)
        arrays['action']=convert_joints(arrays['action'],conversion).astype(np.float32)
        if 'joint_velocity' in arrays:arrays['joint_velocity']=convert_joints(arrays['joint_velocity'],conversion,velocity=True).astype(np.float32)
        np.savez_compressed(directory/'trajectory.npz',**arrays)
        info=json.loads((original/'episode.json').read_text())
        info.update(joint_names=metadata['joint_names'],joint_units=conversion['units'],
            source_episode=str(original.resolve()),source_images={alias:str((original/'images'/camera).resolve()) for camera,alias in conversion['camera_names'].items()})
        (directory/'episode.json').write_text(json.dumps(info,indent=2)+'\n')
        metadata['episodes'].append(dict(index=index,path=str(directory.relative_to(output)),frames=episode['frames'],
            success=episode['success'],complete=episode['complete'],source_path=episode['path']))
    (output/'conversion.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(f'Wrote {len(episodes)} SO101 trajectory array bundles to {output}; images remain at the source. Omit --trajectories-only for the full LeRobot dataset.')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path);parser.add_argument('output',type=Path)
    parser.add_argument('--repo-id',default='local/attempt-codex-pick-place')
    parser.add_argument('--include-failed',action='store_true')
    parser.add_argument('--episode-selection',type=Path,help='JSON list of source manifest episode indices to export')
    parser.add_argument('--allow-assisted',action='store_true',help='explicitly allow legacy datasets that used cube attachment and no physical jaw contacts')
    coordinates=parser.add_mutually_exclusive_group()
    coordinates.add_argument('--joint-map',type=Path,help='JSON with scale[6], offset[6], units; output = scale * sim_radians + offset')
    coordinates.add_argument('--so101',action='store_true',help='Match the reference SO101 coordinates/features: arm degrees, gripper percent, wrist/external cameras')
    parser.add_argument('--so101-joint-map',type=Path,help='Override the CAD-to-recording calibration used by --so101; otherwise use the saved snapshot or current fitted map')
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--validate-only',action='store_true')
    mode.add_argument('--trajectories-only',action='store_true',help='With --so101, write converted NPZ arrays and metadata without LeRobot; this is not the full Parquet/video dataset')
    args=parser.parse_args()
    if args.so101_joint_map and not args.so101:parser.error('--so101-joint-map requires --so101')
    if args.trajectories_only and not args.so101:parser.error('--trajectories-only requires --so101')
    manifest=json.loads((args.source/'dataset.json').read_text())
    if manifest.get('grasp_mode')!='physical_jaw_contacts' and not args.allow_assisted:
        raise ValueError('Legacy assisted-grasp dataset: jaw contacts were not simulated. Recollect for physical-grasp training, '
                         'or explicitly use --allow-assisted for legacy pipeline experiments.')
    fps=float(manifest['fps'])
    if abs(fps-round(fps))>1e-9:raise ValueError('LeRobot export requires an integer FPS')
    episodes=[e for e in manifest['episodes'] if e['frames'] and (args.include_failed or usable(e,args.source))]
    if args.episode_selection:
        selected=json.loads(args.episode_selection.read_text())
        if not isinstance(selected,list) or any(type(i)!=int for i in selected) or len(selected)!=len(set(selected)):
            raise ValueError('Episode selection must be a JSON list of distinct integer source indices')
        eligible={e['index'] for e in episodes}
        if not set(selected)<=eligible:raise ValueError('Selection includes missing or ineligible episodes')
        episodes=[e for e in episodes if e['index'] in set(selected)]
    if not episodes:raise ValueError('No eligible episodes; collect successful complete episodes first')
    scale=np.ones(6);offset=np.zeros(6);units=manifest['joint_units']
    conversion=None;camera_names={camera:camera for camera in manifest['cameras']};joint_names=manifest['joint_names'];robot_type='so101_pychrono'
    if args.so101:
        conversion=so101_conversion(manifest,args.so101_joint_map)
        scale=np.asarray(conversion['scale']);offset=np.asarray(conversion['offset']);units=conversion['units']
        camera_names=conversion['camera_names'];joint_names=conversion['joint_names'];robot_type=conversion['robot_type']
    def transform(values):
        return convert_joints(values,conversion) if conversion else np.asarray(values)*scale+offset
    if args.joint_map:
        mapping=json.loads(args.joint_map.read_text());scale=np.asarray(mapping['scale']);offset=np.asarray(mapping['offset']);units=mapping['units']
        if scale.shape!=(6,) or offset.shape!=(6,) or not np.isfinite(scale).all() or not np.isfinite(offset).all() or (scale==0).any():
            raise ValueError('Joint mapping requires six finite nonzero scales and six finite offsets')
    for episode in episodes:
        directory=args.source/episode['path']
        with np.load(directory/'trajectory.npz') as saved:d={key:saved[key] for key in saved.files}
        n=len(d['timestamp'])
        if n!=episode['frames'] or not np.allclose(d['timestamp'],np.arange(n)/fps,atol=1e-6):raise ValueError('Frame count or timestamps are inconsistent')
        if d['observation_state'].shape!=(n,6) or d['action'].shape!=(n,6):raise ValueError('State/action shape must be (frames,6)')
        if not np.isfinite(d['observation_state']).all() or not np.isfinite(d['action']).all():raise ValueError('Nonfinite joint data')
        if conversion:
            check_gripper_range(transform(d['observation_state']));check_gripper_range(transform(d['action']))
        if np.max(np.ptp(d['image_timestamps'],axis=1))>1e-4:raise ValueError('Camera timestamps are not synchronized')
        if n>1 and not (np.diff(d['image_launches'],axis=0)>0).all():raise ValueError('Repeated or out-of-order images')
        if not args.trajectories_only:
            for camera in manifest['cameras']:
                for i in range(n):
                    if not (directory/'images'/camera/f'{i:06d}.png').is_file():raise ValueError('Missing camera image')
    print(f'Validated {len(episodes)} episodes, {sum(e["frames"] for e in episodes)} frames, {fps:g} FPS; joint units: {units}')
    if conversion:
        print(f'SO101 calibration: {conversion["calibration_origin"]}; cameras: {camera_names}; absolute action targets retain their recorded timing')
    if args.trajectories_only:
        export_trajectories(args.source,args.output,manifest,episodes,conversion)
        return
    if args.validate_only:return
    first=args.source/episodes[0]['path']/'images'/manifest['cameras'][0]/'000000.png'
    with Image.open(first) as image:width,height=image.size
    if args.output.exists():raise FileExistsError(f'Output already exists: {args.output}')
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        raise SystemExit('Install LeRobot in this Python environment to export; --validate-only does not need it.') from exc
    names=[f'{joint}.pos' for joint in joint_names]
    features={key:dict(dtype='float32',shape=(6,),names=names) for key in ['observation.state','action']}
    features.update({f'observation.images.{camera_names[camera]}':dict(dtype='video',shape=(height,width,3),names=['height','width','channels']) for camera in manifest['cameras']})
    dataset=LeRobotDataset.create(repo_id=args.repo_id,root=args.output,fps=round(fps),features=features,
        robot_type=robot_type,use_videos=True,image_writer_threads=2,vcodec='h264')
    metadata=[]
    try:
        for episode in episodes:
            directory=args.source/episode['path']
            with np.load(directory/'trajectory.npz') as saved:d={key:saved[key] for key in saved.files}
            state=transform(d['observation_state']).astype(np.float32);action=transform(d['action']).astype(np.float32)
            info=json.loads((directory/'episode.json').read_text())
            for i in range(len(d['timestamp'])):
                frame={'observation.state':state[i],'action':action[i],'task':info['task']}
                frame.update({f'observation.images.{camera_names[camera]}':np.array(Image.open(directory/'images'/camera/f'{i:06d}.png').convert('RGB')) for camera in manifest['cameras']})
                dataset.add_frame(frame)
            dataset.save_episode()
            metadata.append(dict(info,source_episode_path=episode['path'],source_manifest_index=episode.get('index'),
                source_worker_id=episode.get('worker_id'),source_worker_episode_index=episode.get('worker_episode_index')))
            print(f'Exported episode {len(metadata)}/{len(episodes)}: {episode["path"]}',flush=True)
    finally:
        dataset.finalize()
    (args.output/'chrono_metadata.json').write_text(json.dumps(dict(joint_units=units,joint_scale=scale.tolist(),joint_offset=offset.tolist(),
        grasp_mode=manifest['grasp_mode'],action_semantics=manifest['action_semantics'],jaw_collision=manifest.get('jaw_collision'),
        cameras=[camera_names[c] for c in manifest['cameras']],source_camera_names=manifest['cameras'],camera_config=manifest.get('camera_config'),
        so101_conversion=conversion,robot_type=robot_type,
        work_surface=manifest.get('work_surface'),tray_geometry=manifest.get('tray_geometry'),source_episodes=metadata),indent=2)+'\n')
    print('Exported locally to',args.output)

if __name__=='__main__':main()
