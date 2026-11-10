#!/usr/bin/env python3
"""Fit the external optical pose to the recorded box/mat, retaining the lens."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from reference_camera import load_profile, project_world
from tools.fit_reference_camera import landmarks, rotation


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'assets/video_camera.json')
    parser.add_argument('--refine-lens',action='store_true',help='fit independent intrinsics with a prior from the other camera of the same model')
    parser.add_argument('--stereo-plates',action='store_true',help='also fit known-size plates reconstructed from the independently fitted wrist camera')
    args=parser.parse_args()
    previous=load_profile(ROOT/'assets/reference_camera.json')
    # Top-left pixels, measured in external chunk-000/file-000, frame 0.
    # The far/front board corner is partly hidden, so it receives less weight.
    pixels=np.array([[184,153],[357,158],[82,439],[438,450],
                     [209,194],[315,198],[169,348],[331,352]],dtype=float)
    marks=landmarks();world=np.array([v[0] for v in marks.values()])
    weights=np.array([.35,1,1,1,1,1,1,1])
    initial=np.r_[previous['optical_position_m'],previous['yaw_pitch_roll_deg']]
    boundary=None;field=None;seam_field=None;plate_constraints=[]
    if args.refine_lens:
        import cv2
        from scipy.ndimage import distance_transform_edt
        from reference_camera import load_wrist_profile
        from fit_metric_twin import mat_boundary
        reference_lens=load_wrist_profile()['lens']
        lens_initial=np.r_[reference_lens['focal_px'],reference_lens['principal_point_px'],reference_lens['theta_coefficients']]
        initial=np.r_[initial,lens_initial]
        image=cv2.imread(str(ROOT/'output/video_fit/real_external_000.png'),cv2.IMREAD_GRAYSCALE)
        n,labels,stats,_=cv2.connectedComponentsWithStats((image<100).astype(np.uint8))
        mask=(labels==labels[300,250]).astype(np.uint8)
        contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(mask,contours,-1,1,-1)
        field=distance_transform_edt(mask==0)-distance_transform_edt(mask!=0)
        boundary=mat_boundary(45)
        edges=cv2.Canny(image,30,90);edges[:415]=0;edges[:,:75]=0;edges[:,445:]=0
        seam_field=distance_transform_edt(edges==0)
        if args.stereo_plates:
            from reference_objects import fit_initial_objects,object_mask,box_vertices
            from so101_ik import forward_kinematics
            assets=json.loads((ROOT/'assets/scene_assets.json').read_text())
            mapping=json.loads((ROOT/'assets/video_joint_map.json').read_text())
            wrist=load_wrist_profile();directory=ROOT/'output/metric_refit/calibration'
            records=json.loads((directory/'measurements.json').read_text())
            for ep in range(8):
                row=min((r for r in records if r['episode']==ep),key=lambda r:r['timestamp'])
                wi=cv2.cvtColor(cv2.imread(str(directory/f'{row["index"]:06d}_wrist.png')),cv2.COLOR_BGR2RGB)
                ei=cv2.cvtColor(cv2.imread(str(directory/f'{row["index"]:06d}_external.png')),cv2.COLOR_BGR2RGB)
                q=np.array(row['state'])*mapping['scale']+mapping['offset_deg']
                pose=forward_kinematics(assets,q)[5]@np.array(wrist['link_from_optical'])
                wp=dict(lens=wrist['lens'],optical_position_m=pose[:3,3],world_from_optical_rotation=pose[:3,:3])
                plate=fit_initial_objects({'wrist':wi},{'wrist':wp})['plate']
                target=object_mask(ei,False)[0]
                contours,_=cv2.findContours(target,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(target,contours,-1,1,-1)
                signed=distance_transform_edt(target==0)-distance_transform_edt(target!=0)
                corners=box_vertices(plate['center_m'],plate['yaw_rad'],plate['dimensions_m'])
                plate_constraints.append((corners,signed))
    def profile(p):
        result={**previous,'optical_position_m':p[:3].tolist(),
                'world_from_optical_rotation':rotation(*p[3:6]).tolist()}
        if args.refine_lens:
            result['lens']=dict(model='equidistant_polynomial',focal_px=float(p[6]),principal_point_px=p[7:9].tolist(),theta_coefficients=p[9:11].tolist())
        return result
    def residual(p):
        measured=(project_world(world,profile(p))-pixels)*weights[:,None]*(2.5 if args.refine_lens else 1.)
        result=np.r_[measured.ravel(),(p[:3]-initial[:3])/.06,(p[3:6]-initial[3:6])/15]
        if args.refine_lens:
            from scipy.ndimage import map_coordinates
            uv=project_world(boundary,profile(p));valid=(uv[:,0]>218)|(uv[:,1]>285)
            edge=map_coordinates(field,[uv[:,1],uv[:,0]],order=1,mode='nearest')*valid
            floor=np.column_stack([np.full(90,.4445),np.linspace(0,.6096,90),np.zeros(90)])
            uv=project_world(floor,profile(p));seam=map_coordinates(seam_field,[uv[:,1],uv[:,0]],order=1,mode='nearest')
            result=np.r_[result,edge*.6,seam*.35,(p[6:]-lens_initial)/[14,12,12,.04,.025]]
            if plate_constraints:
                import cv2
                for corners,distances in plate_constraints:
                    uv=project_world(corners,profile(p));hull=cv2.convexHull(uv.astype(np.float32)).reshape(-1,2)
                    points=np.concatenate([a+(b-a)*np.linspace(0,1,6)[:,None] for a,b in zip(hull,np.roll(hull,-1,axis=0))])
                    # Eight vertices form a six-sided box hull in these views.
                    # Interpolate to a fixed residual length for solver stability.
                    u=np.linspace(0,len(points)-1,32)
                    points=np.column_stack([np.interp(u,np.arange(len(points)),points[:,j]) for j in [0,1]])
                    measured=map_coordinates(distances,[points[:,1],points[:,0]],order=1,mode='nearest')
                    result=np.r_[result,measured*.6]
            # The radial function must have a unique inverse for every sensor
            # pixel, including corners not used by the landmark fit.
            theta=np.linspace(0,1.95,32)
            derivative=1+3*p[9]*theta**2+5*p[10]*theta**4
            corners=np.array([[0.,0.],[640.,0.],[0.,480.],[640.,480.]])
            needed=np.max(np.linalg.norm((corners-p[7:9])/p[6],axis=1))
            extent=1.95*(1+p[9]*1.95**2+p[10]*1.95**4)
            result=np.r_[result,np.maximum(0,.15-derivative)*500,max(0,needed+.03-extent)*500]
        return result
    if args.refine_lens:
        low=np.r_[[.36,.25,.44],[-5,25,-20],[245,300,210,-.20,-.04]]
        high=np.r_[[.44,.44,.507],[25,60,5],[330,345,265,.10,.08]]
    else:low,high=initial-np.r_[[.06]*3,[12]*3],initial+np.r_[[.06]*3,[12]*3]
    fit=least_squares(residual,initial,x_scale='jac',bounds=(low,high),max_nfev=300,
        loss='soft_l1',f_scale=1.5)
    result=profile(fit.x);predicted=project_world(world,result)
    errors=np.linalg.norm(predicted-pixels,axis=1)
    result.update(source='green_cube_red_plate_1/videos/observation.images.external/chunk-000/file-000.mp4, frame 0',
        method='independent optical pose/lens fit to mat, box corners and curved near-wall seam' if args.refine_lens else 'six-parameter optical pose fit to video floor/mat landmarks; lens held fixed',
        yaw_pitch_roll_deg=fit.x[3:6].tolist(),lens_source=previous['source'],
        landmark_rms_px=float(np.sqrt(np.mean(errors**2))),
        pose_change_from_photo_mm=((fit.x[:3]-initial[:3])*1000).tolist(),
        landmarks={name:dict(world_m=world[i].tolist(),reference_px=pixels[i].tolist(),
            projected_px=predicted[i].tolist(),error_px=float(errors[i]),weight=float(weights[i]))
            for i,name in enumerate(marks)},
        uncertainties=['Hand-located corners, approximately 2 px; partly occluded front corner downweighted',
            'Interior and mat dimensions retained; optical intrinsics held fixed to avoid single-plane overfitting',
            'Pixel alignment estimate, not a measured metric camera calibration'])
    if args.refine_lens:
        result['lens_source']='independent external-lens fit; wrist-camera estimate used only as a weak same-model prior'
        result['uncertainties'][1]='Metric enclosure/mat dimensions and optional known-size stereo plates; optics/pose remain image-derived'
        result['stereo_plate_training_episodes']=list(range(8)) if args.stereo_plates else []
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['landmarks','uncertainties']},indent=2))


if __name__=='__main__':main()
