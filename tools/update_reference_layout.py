#!/usr/bin/env python3
"""Refresh the static default from episode zero after applying calibration."""
import json
from pathlib import Path
import sys
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from reference_camera import load_profile,load_wrist_profile
from so101_ik import forward_kinematics
from reference_objects import fit_initial_objects
from replay_reference import read_video


def main():
    data=np.load(ROOT/'output/video_fit/reference_trajectory.npz');meta=json.loads(str(data['metadata']));ep=meta['episodes'][0]
    mapping=json.loads((ROOT/'assets/video_joint_map.json').read_text());assets=json.loads((ROOT/'assets/scene_assets.json').read_text())
    q=data['observation_state'][0]*mapping['scale']+mapping['offset_deg']
    wrist=load_wrist_profile();pose=forward_kinematics(assets,q)[5]@np.array(wrist['link_from_optical'])
    profiles=dict(external=load_profile(),wrist=dict(lens=wrist['lens'],optical_position_m=pose[:3,3],world_from_optical_rotation=pose[:3,:3]))
    images={}
    for name,video in ep['videos'].items():
        cap=cv2.VideoCapture(str(ROOT.parent/'ref_data/green_cube_red_plate_1'/video['file']))
        images[name]=read_video(cap,round(video['from_timestamp']*meta['fps']));cap.release()
    initial=fit_initial_objects(images,profiles)
    report=dict(method='initial recorded state and two-camera rigid object silhouette fit',source='green_cube_red_plate_1 episode 0 frame 0',
        tray_center_xy_m=initial['plate']['center_m'][:2],tray_yaw_rad=initial['plate']['yaw_rad'],
        cube_center_xy_m=initial['cube']['center_m'][:2],cube_center_xyz_m=initial['cube']['center_m'],cube_yaw_rad=initial['cube']['yaw_rad'],
        inspection_joint_deg=q.tolist(),initial_object_fit=initial,
        assumptions=['Shared fitted base and independent wrist/global optics',
            'Recorded hardware normalization; image-fitted CAD joint zeros',
            'Known 30 mm cube and 50 mm plate; cube initially on the 1 mm mat'])
    (ROOT/'assets/reference_layout.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
