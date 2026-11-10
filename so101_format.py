"""Convert recorded CAD joint coordinates to the supplied LeRobot SO101 convention.

The fitted map is CAD_degrees = real_recorded_value * scale + offset_deg.
Its inverse yields degrees for joints 1--5 and calibrated gripper percentage.
It changes coordinates only; it does not reproduce human demonstration paths.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from motor_calibration import NAMES

ROOT=Path(__file__).resolve().parent
DEFAULT_JOINT_MAP=ROOT/'assets/video_joint_map.json'
SO101_UNITS='so101_arm_degrees_gripper_percent'


def joint_map_snapshot(path=DEFAULT_JOINT_MAP):
    path=Path(path);raw=path.read_bytes();mapping=json.loads(raw)
    scale=np.asarray(mapping['scale'],dtype=float);offset=np.asarray(mapping['offset_deg'],dtype=float)
    if scale.shape!=(6,) or offset.shape!=(6,) or not np.isfinite(scale).all() or not np.isfinite(offset).all() or (scale==0).any():
        raise ValueError('SO101 CAD map needs six finite nonzero scales and six finite offsets in degrees')
    return dict(scale=scale.tolist(),offset_deg=offset.tolist(),
        convention='cad_degrees = recorded_so101 * scale + offset_deg',
        joint_names=list(NAMES),source_file=path.name,source_sha256=hashlib.sha256(raw).hexdigest(),
        calibration_method=mapping.get('method'),source=mapping.get('source'))


def so101_conversion(manifest,path=None):
    if manifest['joint_units']!='radians_offset_from_STEP_pose':
        raise ValueError('--so101 expects native CAD-relative radians; refusing to convert already normalized data')
    names=[name.removesuffix('.pos') for name in manifest['joint_names']]
    if len(names)!=6 or set(names)!=set(NAMES):raise ValueError('Recording must name exactly the six SO101 joints')
    if path is not None:
        mapping=joint_map_snapshot(path);origin='explicit_joint_map'
    elif 'so101_joint_map' in manifest:
        mapping=manifest['so101_joint_map'];origin='recorded_calibration_snapshot'
    else:
        mapping=joint_map_snapshot();origin='current_joint_map_for_legacy_recording'
    scale=np.asarray(mapping['scale'],dtype=float);offset=np.asarray(mapping['offset_deg'],dtype=float)
    if scale.shape!=(6,) or offset.shape!=(6,) or not np.isfinite(scale).all() or not np.isfinite(offset).all() or (scale==0).any():
        raise ValueError('Invalid recorded SO101 calibration snapshot')
    cameras=manifest['cameras']
    if len(cameras)!=2 or set(cameras)!={'wrist','right_45'}:
        raise ValueError('--so101 requires wrist and right_45 cameras; an overhead view cannot be relabeled as the real external camera')
    return dict(joint_names=list(NAMES),column_indices=[names.index(name) for name in NAMES],
        scale=((180/np.pi)/scale).tolist(),offset=(-offset/scale).tolist(),
        units=SO101_UNITS,per_joint_units=['degrees']*5+['percent'],robot_type='so_follower',
        camera_names=dict(wrist='wrist',right_45='external'),
        calibration_origin=origin,cad_joint_map=mapping,
        equation='so101 = (rad2deg(native_cad_radians) - cad_offset_deg) / cad_scale',
        notes=['Hardware firmware homing offsets are already represented; do not apply them a second time',
            'Observations remain measured positions and actions remain the recorded absolute next-tick targets',
            'Timestamps, sampling rate and image resolution are preserved; no action-index shift or clipping',
            'Coordinate conversion does not make the synthetic IK paths identical to recorded human motions'])


def convert_joints(values,conversion,velocity=False):
    values=np.asarray(values,dtype=float)
    if values.ndim==0 or values.shape[-1]!=6 or not np.isfinite(values).all():raise ValueError('Expected finite six-joint values')
    converted=values[...,conversion['column_indices']]*conversion['scale']
    if not velocity:converted=converted+conversion['offset']
    return converted


def check_gripper_range(values):
    gripper=np.asarray(values)[...,5]
    if np.any((gripper<-.0001)|(gripper>100.0001)):
        raise ValueError(f'Converted gripper is outside 0--100 ({gripper.min():.5g} to {gripper.max():.5g}); check the recording calibration; values are not silently clipped')
