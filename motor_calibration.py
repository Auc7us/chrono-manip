"""Offline LeRobot/STS3215 position conversions; never accesses a motor bus.

Present_Position and calibration ranges already include the firmware homing
offset. Applying homing_offset again to a saved observation double counts it.
CAD assembly zeros remain a separate geometric calibration.
"""
import numpy as np

NAMES=('shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll','gripper')
SIGNS=np.array([-1.,1.,1.,1.,-1.,-1.])
DEG_PER_TICK=360./4095.


def validate(calibration):
    if set(calibration)!=set(NAMES):raise ValueError('Expected the six SO101 motors')
    for i,name in enumerate(NAMES):
        motor=calibration[name]
        if motor['id']!=i+1:raise ValueError(f'Unexpected motor ID for {name}')
        if not 0<=motor['range_min']<motor['range_max']<=4095:raise ValueError(f'Invalid range for {name}')
        if motor['drive_mode'] not in (0,1):raise ValueError(f'Invalid drive mode for {name}')


def to_ticks(values,calibration,body_mode='degrees',command=False):
    """Invert normalization. Commands use the SDK's int() truncation/clipping."""
    validate(calibration);values=np.asarray(values,dtype=float)
    if values.shape[-1]!=6 or not np.isfinite(values).all():raise ValueError('Expected finite six-joint values')
    result=np.empty_like(values)
    for j,name in enumerate(NAMES):
        motor=calibration[name];lo,hi=motor['range_min'],motor['range_max'];val=values[...,j]
        if j<5 and body_mode=='degrees':result[...,j]=val*4095./360.+(lo+hi)/2
        else:
            if j<5:
                if body_mode!='range_m100_100':raise ValueError(body_mode)
                if motor['drive_mode']:val=-val
                if command:val=np.clip(val,-100.,100.)
                fraction=(val+100.)/200.
            else:
                if motor['drive_mode']:val=100.-val
                if command:val=np.clip(val,0.,100.)
                fraction=val/100.
            result[...,j]=lo+fraction*(hi-lo)
    return np.trunc(result).astype(np.int64) if command else result


def from_ticks(ticks,calibration,body_mode='degrees'):
    validate(calibration);ticks=np.asarray(ticks,dtype=float);result=np.empty_like(ticks)
    for j,name in enumerate(NAMES):
        motor=calibration[name];lo,hi=motor['range_min'],motor['range_max'];val=ticks[...,j]
        if j<5 and body_mode=='degrees':result[...,j]=(val-(lo+hi)/2)*360./4095.
        else:
            fraction=(np.clip(val,lo,hi)-lo)/(hi-lo)
            normalized=fraction*(200. if j<5 else 100.)-(100. if j<5 else 0.)
            if motor['drive_mode']:normalized=-normalized if j<5 else 100.-normalized
            result[...,j]=normalized
    return result


def quantize_commands(values,profile,recover_leader_grid=False):
    """Leader-space saved targets become follower-space integer goal positions."""
    if recover_leader_grid:
        # Saved float32 actions can land just below an integer tick. Recover
        # the validated leader readings before reproducing SDK arithmetic.
        leader=profile['leader']['calibration'];ticks=to_ticks(values,leader)
        if abs(ticks-np.rint(ticks)).max()>.001:raise ValueError('Saved commands do not lie on the supplied leader encoder grid')
        values=from_ticks(np.rint(ticks),leader)
    calibration=profile['follower']['calibration']
    return from_ticks(to_ticks(values,calibration,command=True),calibration)


def nominal_home_offsets(calibration,home_tick=2048):
    """Approximate CAD reference, conditional on the hand-set calibration pose.

    The JSON stores encoder ranges, not a surveyed physical home pose. These
    offsets are useful priors, never a claim of exact joint-zero measurements.
    """
    home=from_ticks(np.full(6,float(home_tick)),calibration)
    return np.array([0.,90.,-90.,0.,90.])-SIGNS[:5]*home[:5]
