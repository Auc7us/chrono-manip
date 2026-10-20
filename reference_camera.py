"""Nominal/reference camera poses and direct OptiX fisheye rays.

Only this demo's process selects the local shader directory. Installed Chrono
libraries and shaders are never edited. Both live feeds and recordings use the
same actual ray projection, without warping saved images afterward.
"""
from pathlib import Path
import hashlib
import fcntl
import json
import os
import shutil
import subprocess
import numpy as np

ROOT=Path(__file__).resolve().parent
DEFAULT_PROFILE=ROOT/'assets/nominal_camera.json'
PHOTO_FIT_PROFILE=ROOT/'assets/reference_camera.json'


def load_profile(path=None):
    return json.loads(Path(path or DEFAULT_PROFILE).read_text())


def fitted_board_pose(profile=None,optical_offset=.019):
    profile=profile or load_profile()
    r=np.array(profile['world_from_optical_rotation'])
    result=np.eye(4);result[:3,:3]=r
    result[:3,3]=np.array(profile['optical_position_m'])-r[:,0]*optical_offset
    return result


def pixel_rays(pixels,profile=None):
    profile=profile or load_profile();lens=profile['lens']
    xy=(np.asarray(pixels)-lens['principal_point_px'])/lens['focal_px']
    radius=np.linalg.norm(xy,axis=1);theta=radius.copy();k1,k2=lens['theta_coefficients']
    for _ in range(10):
        t2=theta*theta;theta-=(theta*(1+k1*t2+k2*t2*t2)-radius)/(1+3*k1*t2+5*k2*t2*t2)
    scale=np.sin(theta)/np.maximum(radius,1e-12)
    return np.column_stack([np.cos(theta),-scale[:,None]*xy])@np.array(profile['world_from_optical_rotation']).T


def project_world(points,profile=None):
    profile=profile or load_profile();lens=profile['lens']
    local=(np.asarray(points)-profile['optical_position_m'])@np.array(profile['world_from_optical_rotation'])
    xy=-local[:,1:];radius=np.linalg.norm(xy,axis=1);theta=np.arctan2(radius,local[:,0])
    k1,k2=lens['theta_coefficients'];rd=theta*(1+k1*theta**2+k2*theta**4)
    return lens['focal_px']*xy/np.maximum(radius[:,None],1e-12)*rd[:,None]+lens['principal_point_px']


def configure_projection(sensor_module,profile):
    """Compile/cache the small ray shader and select its private directory."""
    source=ROOT/'assets/shaders/camera_raygen.cu'
    installed=Path(sensor_module.GetSensorShaderDir()).resolve()
    # Other demo instances may have configured this module already in-process.
    if (installed/'installed_directory.txt').is_file():
        installed=Path((installed/'installed_directory.txt').read_text())
    lens=profile['lens']
    parameters=dict(REF_FOCAL=lens['focal_px'],REF_CX=lens['principal_point_px'][0],
                    REF_CY=lens['principal_point_px'][1],REF_K1=lens['theta_coefficients'][0],
                    REF_K2=lens['theta_coefficients'][1])
    from scene_config import WIDTH,DEPTH
    parameters['FLOOR_PATCH_RADIUS_SQ']=WIDTH*DEPTH/(8*6*np.pi)
    parameters['CEILING_PATCH_RADIUS_SQ']=WIDTH*(DEPTH-.16)/(6*4*np.pi)
    material_source=source.with_name('material_shaders.cu')
    spot_header=source.parent/'chrono_sensor/optix/shaders/ChOptixSpotLight.cuh'
    signature=hashlib.sha256(source.read_bytes()+material_source.read_bytes()+spot_header.read_bytes()+json.dumps(parameters,sort_keys=True).encode()+
                            (installed/'camera_raygen.ptx').read_bytes()).hexdigest()[:16]
    cache=ROOT/'output/shader_cache'/signature;cache.mkdir(parents=True,exist_ok=True)
    with (cache/'build.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        prefix=installed.parents[2]
        nvcc=shutil.which('nvcc')
        optix=Path(os.environ.get('OptiX_INSTALL_DIR',ROOT/'../../packages/optix')).resolve()
        for name,shader_source in [('camera_raygen',source),('material_shaders',material_source)]:
            target=cache/(name+'.ptx')
            if target.is_file():continue
            if not nvcc:raise RuntimeError('The local camera/panel shaders need nvcc for their first launch in the Chrono container.')
            temporary=cache/f'{name}.{os.getpid()}.ptx'
            command=[nvcc,'--ptx','--std=c++17','--gpu-architecture=compute_89',
                     '-I'+str(source.parent),'-I'+str(prefix/'include'),'-I'+str(optix/'include'),
                     str(shader_source),'-o',str(temporary)]
            command += [f'-D{k}={v:.12e}f' for k,v in parameters.items()]
            result=subprocess.run(command,capture_output=True,text=True)
            if result.returncode:
                temporary.unlink(missing_ok=True)
                raise RuntimeError('Cannot compile the local '+name+' shader:\n'+result.stderr)
            temporary.replace(target)
    for shader in installed.glob('*.ptx'):
        destination=cache/shader.name
        if shader.name not in ['camera_raygen.ptx','material_shaders.ptx'] and not destination.exists():
            try:destination.symlink_to(shader)
            except FileExistsError:pass
    (cache/'installed_directory.txt').write_text(str(installed))
    sensor_module.SetSensorShaderDir(str(cache))
    return signature


def camera_metadata(args):
    profile=load_profile(args.camera_profile)
    lens=profile['lens'];cx,cy=lens['principal_point_px']
    left,right=pixel_rays([[0,cy],[640,cy]],profile)
    hfov=np.rad2deg(np.arccos(np.clip(left@right,-1,1)))
    return dict(projection=args.lens_model,reference_profile=profile['source'],
                camera_profile_file=str(args.camera_profile),
                lens_source=profile.get('lens_source',profile['source']),
                fitted_lens=profile['lens'] if args.lens_model=='reference' else None,
                effective_hfov_deg=float(hfov) if args.lens_model=='reference' else args.hfov_deg,
                output_focal_px=lens['focal_px']*args.width/640 if args.lens_model=='reference' else None,
                output_focal_xy_px=[lens['focal_px']*args.width/640,lens['focal_px']*(args.width*3//4)/480] if args.lens_model=='reference' else None,
                output_principal_point_px=[cx*args.width/640,cy*(args.width*3//4)/480] if args.lens_model=='reference' else None,
                response='raw' if args.raw_response else 'fixed estimated per-camera ISP response',
                fixed_reference_landmark_rms_px=profile.get('landmark_rms_px'),
                estimated_pose_landmark_rms_px=profile.get('landmark_rms_px') if args.lens_model=='reference' and args.global_camera=='right-45' else None,
                calibration_kind=profile['method'])
