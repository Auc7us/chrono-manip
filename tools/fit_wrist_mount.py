#!/usr/bin/env python3
"""Estimate a small wrist optical-frame correction from fixed CAD features."""
from pathlib import Path
import json
import sys
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from reference_camera import PHOTO_FIT_PROFILE,load_profile,project_world
from so101_ik import forward_kinematics,APPROACH


def main():
    assets=json.loads((ROOT/'assets/scene_assets.json').read_text())
    layout=json.loads((ROOT/'assets/reference_layout.json').read_text())
    frames=forward_kinematics(assets,layout['inspection_joint_deg']);cam=assets['wrist_camera']
    board=np.eye(4);board[:3,:3]=cam['rotation'];board[:3,3]=cam['board_center_m']
    optical=np.eye(4);optical[:3,:3]=np.diag([1.,-1.,-1.]);optical[0,3]=cam['optical_offset_m']
    native=frames[5]@board@optical
    vertices=np.array([[float(x) for x in line.split()[1:4]]
        for line in (ROOT/'assets/meshes/fixed_jaw_collision.obj').read_text().splitlines() if line.startswith('v ')])
    along=vertices@APPROACH;tip=vertices[along>along.max()-.0006].mean(0)
    # The adapter is above the older STEP screw heads. Fit the visible replacement
    # heads on its 35 mm outer face, rather than the occluded original CAD heads.
    features=np.array([[.03615,-.14548569,.14720039],
                       [.03615,-.14555698,.15529884],tip])
    world=features@frames[5][:3,:3].T+frames[5][:3,3]
    target=np.array([[282,433],[333,432],[299,293]])
    profile=load_profile(PHOTO_FIT_PROFILE)
    def frame(p):
        correction=np.eye(4);correction[:3,:3]=Rotation.from_rotvec(np.deg2rad(p[3:])).as_matrix()
        correction[:3,3]=p[:3]*.001
        return native@correction
    def projection(p):
        t=frame(p)
        return project_world(world,{**profile,'optical_position_m':t[:3,3],
                                    'world_from_optical_rotation':t[:3,:3]})
    fit=least_squares(lambda p:np.r_[(projection(p)-target).ravel(),p[:3]/12,p[3:]/18],
        np.zeros(6),bounds=([-12]*3+[-18]*3,[12]*3+[18]*3),max_nfev=1000,x_scale='jac')
    correction=np.linalg.inv(native)@frame(fit.x)
    moving=np.array([[float(x) for x in line.split()[1:4]]
        for line in (ROOT/'assets/meshes/moving_jaw_collision.obj').read_text().splitlines() if line.startswith('v ')])
    along=moving@APPROACH;moving_tip=moving[along>along.max()-.0006].mean(0)
    def moving_pixel(deg):
        q=layout['inspection_joint_deg'].copy();q[5]=float(deg[0]);f=forward_kinematics(assets,q)[6]
        world_tip=f[:3,:3]@moving_tip+f[:3,3];pose=frame(fit.x)
        return project_world(world_tip[None,:],{**profile,'optical_position_m':pose[:3,3],
                             'world_from_optical_rotation':pose[:3,:3]})[0]
    jaw=least_squares(lambda q:(moving_pixel(q)-[337,293]),[80.],bounds=([45.],[108.]))
    layout['inspection_joint_deg'][5]=float(jaw.x[0])
    (ROOT/'assets/reference_layout.json').write_text(json.dumps(layout,indent=2)+'\n')
    report=dict(method='fixed jaw CAD screw centers and tip alignment; bounded optical correction',
        optical_from_corrected_optical=correction.tolist(),translation_mm=fit.x[:3].tolist(),
        rotation_vector_deg=fit.x[3:].tolist(),
        feature_rms_px=float(np.sqrt(np.mean(np.sum((projection(fit.x)-target)**2,axis=1)))),
        reference_pixels=target.tolist(),projected_pixels=projection(fit.x).tolist(),
        fixed_link_points_m=features.tolist(),
        static_jaw_deg=float(jaw.x[0]),moving_tip_projected_px=moving_pixel(jaw.x).tolist(),
        assumptions=['Small optical alignment correction; mechanical mount mesh remains the supplied STL',
                     'Three hand-located features; estimate is not a measured lens calibration'])
    (ROOT/'assets/reference_wrist.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
