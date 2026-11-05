#!/usr/bin/env python3
"""Jointly fit the motor map, base placement and wrist optics across episodes.

Uses the known mat rectangle, fixed CAD jaw features, moving jaw tip and global
arm silhouette. Episodes 8/9 are reserved for validation. All image measurements
and before/after residuals are saved; no frame-specific robot correction is used.
"""
import argparse
import json
import sys
from pathlib import Path
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt,map_coordinates
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from reference_camera import load_profile,project_world
from so101_ik import APPROACH


def mat_boundary(n=64):
    # Same 7 mm rounded rectangle as the actual scene mesh.
    points=[];w,d,r=.4572,.3048,.007
    for cx,cy,start in [(w/2-r,d/2-r,0),(-w/2+r,d/2-r,90),
                        (-w/2+r,-d/2+r,180),(w/2-r,-d/2+r,270)]:
        for angle in np.deg2rad(np.linspace(start,start+90,12)):
            points.append([cx+r*np.cos(angle),cy+r*np.sin(angle)+.3048,.001])
        # Straight sections are filled below from the exact tangent endpoints.
    corners=np.array([[-w/2+r,-d/2], [w/2-r,-d/2], [w/2,-d/2+r],
        [w/2,d/2-r],[w/2-r,d/2],[-w/2+r,d/2],[-w/2,d/2-r],[-w/2,-d/2+r]])
    for i in [0,2,4,6]:
        a,b=corners[i],corners[i+1]
        for xy in np.linspace(a,b,n):points.append([xy[0],xy[1]+.3048,.001])
    return np.array(points)


def segmentation(image):
    gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    n,labels,stats,_=cv2.connectedComponentsWithStats((gray<115).astype(np.uint8))
    candidates=[i for i in range(1,n) if stats[i,cv2.CC_STAT_AREA]>14000 and
                np.count_nonzero(labels[65:275,145:510]==i)>5000]
    if not candidates:return None
    label=max(candidates,key=lambda i:stats[i,cv2.CC_STAT_AREA])
    contours,_=cv2.findContours((labels==label).astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    mask=np.zeros_like(gray);cv2.drawContours(mask,contours,-1,1,-1)
    signed=distance_transform_edt(mask==0)-distance_transform_edt(mask!=0)
    return mask,signed


def moving_tip(image):
    r,g,b=image[:,:,0].astype(float),image[:,:,1].astype(float),image[:,:,2].astype(float)
    white=(r>145)&(g>145)&(b>115)
    roi=white[275:365,316:490].astype(np.uint8)
    n,labels,stats,_=cv2.connectedComponentsWithStats(roi)
    candidates=[i for i in range(1,n) if stats[i,cv2.CC_STAT_AREA]>100 and stats[i,cv2.CC_STAT_TOP]<45]
    if not candidates:return None
    label=max(candidates,key=lambda i:stats[i,cv2.CC_STAT_AREA]);y,x=np.nonzero(labels==label)
    top=y.min();keep=y<top+5
    return [float(np.median(x[keep])+316),float(np.median(y[keep])+275)]


def prepare(trajectory,dataset,output):
    from replay_reference import read_video
    data=np.load(trajectory);meta=json.loads(str(data['metadata']));records=[]
    for ep in meta['episodes']:
        indices=np.flatnonzero(data['episode_index']==ep['episode_index'])
        selected=np.unique(np.r_[indices[np.linspace(0,len(indices)-1,8,dtype=int)],
            indices[np.argmin(data['observation_state'][indices,:5],axis=0)],
            indices[np.argmax(data['observation_state'][indices,:5],axis=0)]]).astype(int)
        caps={name:cv2.VideoCapture(str(dataset/v['file'])) for name,v in ep['videos'].items()}
        for index in selected:
            row=dict(index=int(index),episode=ep['episode_index'],state=data['observation_state'][index].tolist(),
                     timestamp=float(data['timestamp'][index]),training=ep['episode_index']<8)
            images={name:read_video(cap,round((ep['videos'][name]['from_timestamp']+row['timestamp'])*meta['fps']))
                    for name,cap in caps.items()}
            segment=segmentation(images['wrist'])
            if segment is None:continue
            row['moving_tip_px']=moving_tip(images['wrist'])
            records.append(row)
            for name,image in images.items():
                Image=__import__('PIL.Image',fromlist=['Image'])
                Image.fromarray(image).save(output/f'{index:06d}_{name}.png')
        for cap in caps.values():cap.release()
    (output/'measurements.json').write_text(json.dumps(records,indent=2)+'\n')
    print(f'Prepared {len(records)} paired views; {sum(r["training"] for r in records)} training, others held out',flush=True)


def axis_rotations(axes,angles):
    # angles is N x joints, axes is joints x 3.
    x,y,z=axes.T;k=np.zeros((len(axes),3,3));k[:,0,1]=-z;k[:,0,2]=y;k[:,1,0]=z;k[:,1,2]=-x;k[:,2,0]=-y;k[:,2,1]=x
    return np.eye(3)+np.sin(angles)[:,:,None,None]*k[None]+(1-np.cos(angles))[:,:,None,None]*(k@k)[None]


def fit(a):
    baseline=a.output.parent/'before_assets'
    if not (baseline/'video_joint_map.json').exists():baseline=ROOT/'assets'
    assets=json.loads((ROOT/'assets/scene_assets.json').read_text());old_map=json.loads((baseline/'video_joint_map.json').read_text())
    old_wrist=json.loads((baseline/'reference_wrist.json').read_text())
    baseline_external=load_profile(baseline/'video_camera.json');external=load_profile()
    cam=assets['wrist_camera'];board=np.eye(4);board[:3,:3]=cam['rotation'];board[:3,3]=cam['board_center_m']
    optical=np.diag([1.,-1.,-1.,1.]);optical[0,3]=cam['optical_offset_m']
    local=board@optical@np.array(old_wrist['optical_from_corrected_optical'])
    records=json.loads((a.output/'measurements.json').read_text());states=np.array([r['state'] for r in records]);n=len(states)
    pivots=np.array([j['pivot_m'] for j in assets['joints']]);axes=np.array([j['axis'] for j in assets['joints']]);axes/=np.linalg.norm(axes,axis=1)[:,None]
    boundary=mat_boundary(40)
    fields=[];global_fields=[];plate_fields=[];occlusion_fields=[]
    from reference_objects import object_mask,fit_initial_objects
    for row in records:
        wrist=cv2.cvtColor(cv2.imread(str(a.output/f'{row["index"]:06d}_wrist.png')),cv2.COLOR_BGR2RGB)
        fields.append(segmentation(wrist)[1])
        red=object_mask(wrist,False);green=object_mask(wrist,True)
        plate_fields.append(None if red is None else distance_transform_edt(red[0]==0)-distance_transform_edt(red[0]!=0))
        occlusion_fields.append(np.zeros((480,640)) if green is None else cv2.dilate(green[0],np.ones((9,9),np.uint8)).astype(float))
        im=cv2.cvtColor(cv2.imread(str(a.output/f'{row["index"]:06d}_external.png')),cv2.COLOR_BGR2RGB).astype(float)
        mask=(im[:,:,0]-im[:,:,2]>12)&(im[:,:,1]-im[:,:,2]>9)&(im[:,:,0]>80)&(im[:,:,1]>80)&(im[:,:,1]<im[:,:,0]*1.15)
        mask[:125]=False;mask[:,330:]=False;mask[310:]=False
        mask=cv2.dilate(mask.astype(np.uint8),np.ones((7,7),np.uint8))
        global_fields.append(distance_transform_edt(mask==0))
    # Sparse cream surface samples are only a broad global silhouette constraint.
    surfaces=[]
    for part in assets['meshes']:
        if part['material']!='cream' or part['link']>4:continue
        vertices=np.array([[float(x) for x in line.split()[1:4]] for line in (ROOT/'assets/meshes'/part['file']).read_text().splitlines() if line.startswith('v ')])
        surfaces.append((part['link'],vertices[::max(1,len(vertices)//65)]))
    jaw=json.loads((ROOT/'assets/reference_wrist.json').read_text())['fixed_link_points_m']
    fixed_features=np.array(jaw);fixed_targets=np.array([[283.,430.],[334.,430.],[299.,291.]])
    moving=np.array([[float(x) for x in line.split()[1:4]] for line in (ROOT/'assets/meshes/moving_jaw_collision.obj').read_text().splitlines() if line.startswith('v ')])
    along=moving@APPROACH;moving=moving[along>along.max()-.0006].mean(0)
    # Parameters: scales(5), offsets(5), base XYZ mm(3), optical delta XYZ mm /
    # rotation deg(6), independent wrist lens f,cx,cy,k1,k2(5), jaw scale/offset(2).
    lens=baseline_external['lens'];p0=np.r_[old_map['scale'][:5],old_map['offset_deg'][:5],
        [0.]*9,lens['focal_px'],lens['principal_point_px'],lens['theta_coefficients'],old_map['scale'][5],old_map['offset_deg'][5]]
    lower=np.r_[[-1.4,.55,.55,.55,-1.5],[-35,55,-135,-35,35],[-55,-80,-5],[-10]*3,[-10]*3,
                  [230,300,185,-.25,-.08],[-1.8,80]]
    # The CAD base feet already lie on the measured floor. A fit must not bury
    # the robot to compensate for an optical error.
    lower[12]=-.15;upper_z=.15
    upper=np.r_[[-.2,1.4,1.4,1.4,-.2],[35,125,-55,35,115],[55,40,8],[10]*3,[10]*3,
                  [380,350,270,.20,.15],[-.4,110]]
    upper[12]=upper_z
    encoder_scales=None;hardware=None;home_offsets=None
    if a.motor_calibration:
        hardware=json.loads(a.motor_calibration.read_text())
        home_offsets=np.array(hardware['nominal_home_offset_deg'])
    if a.encoder_scales or hardware is not None:
        encoder_scales=np.array(hardware['signed_scale'] if hardware is not None else json.loads(a.encoder_scales.read_text())['signed_scale'])
        lower[:5]=encoder_scales[:5]-.001;upper[:5]=encoder_scales[:5]+.001
        lower[24]=encoder_scales[5]-.002;upper[24]=encoder_scales[5]+.002
    start=p0.copy();plate_layouts=None;plate_external=load_profile()
    if a.initial_fit:
        start=np.array(json.loads(a.initial_fit.read_text())['parameters'][:26])
    if a.plate_constraints:
        from so101_ik import forward_kinematics
        from reference_camera import load_wrist_profile
        wrist=load_wrist_profile();mapping=json.loads((ROOT/'assets/video_joint_map.json').read_text())
        plate_layouts=[];external_plate_fields=[]
        for ep in range(10):
            row=min((r for r in records if r['episode']==ep),key=lambda r:r['timestamp'])
            images={cam:cv2.cvtColor(cv2.imread(str(a.output/f'{row["index"]:06d}_{cam}.png')),cv2.COLOR_BGR2RGB) for cam in ['wrist','external']}
            q=np.array(row['state'])*mapping['scale']+mapping['offset_deg']
            pose=forward_kinematics(assets,q)[5]@np.array(wrist['link_from_optical'])
            wp=dict(lens=wrist['lens'],optical_position_m=pose[:3,3],world_from_optical_rotation=pose[:3,:3])
            plate=fit_initial_objects(images,dict(wrist=wp,external=plate_external))['plate']
            plate_layouts.append([*plate['center_m'][:2],plate['yaw_rad']])
            mask=object_mask(images['external'],False)[0]
            external_plate_fields.append(distance_transform_edt(mask==0)-distance_transform_edt(mask!=0))
        plate_layouts=np.array(plate_layouts);extra=plate_layouts[:8].ravel()
        p0=np.r_[p0,extra];start=np.r_[start,extra]
        lower=np.r_[lower,extra-np.tile([.015,.015,.30],8)]
        upper=np.r_[upper,extra+np.tile([.015,.015,.30],8)]
        corners=np.array([[-1,-1],[1,-1],[1,1],[-1,1]])*.025
        plate_boundary=np.concatenate([a+(b-a)*np.linspace(0,1,24)[:,None] for a,b in zip(corners,np.roll(corners,-1,axis=0))])
    train=np.array([r['training'] for r in records]);lookup={r['index']:i for i,r in enumerate(records)}
    anchors=[(0,[[48,344],[187,114],[526,207],[537,463]]),
        (90,[[77,340],[189,44],[568,222],[None,None]]),
        (180,[[92,407],[77,53],[629,220],[500,454]])]
    world_corners=np.array([[-.2286,.1524,.001],[-.2286,.4572,.001],[.2286,.4572,.001],[.2286,.1524,.001]])

    def model(p,source_states=None):
        source_states=states if source_states is None else source_states;count=len(source_states)
        q=np.column_stack([source_states[:,:5]*p[:5]+p[5:10],source_states[:,5]*p[24]+p[25]])
        rotations=axis_rotations(axes,np.deg2rad(q))
        frames=np.tile(np.eye(4),(count,7,1,1));frames[:,0,:3,:3]=np.diag([-1.,-1.,1.]);frames[:,0,:3,3]=[0.,.054,-.00039]+p[10:13]*.001
        for j in range(6):
            t=np.tile(np.eye(4),(count,1,1));t[:,:3,:3]=rotations[:,j];t[:,:3,3]=pivots[j]-np.einsum('nij,j->ni',rotations[:,j],pivots[j])
            frames[:,j+1]=frames[:,j]@t
        delta=np.eye(4);delta[:3,:3]=Rotation.from_rotvec(np.deg2rad(p[16:19])).as_matrix();delta[:3,3]=p[13:16]*.001
        local_corrected=local@delta;poses=frames[:,5]@local_corrected
        return frames,poses,local_corrected

    def project(points,poses,p):
        # points can N x V x 3 or V x 3.
        if np.asarray(points).ndim==2:points=np.broadcast_to(points,(len(poses),*points.shape))
        xyz=np.einsum('nvi,nij->nvj',points-poses[:,None,:3,3],poses[:,:3,:3])
        xy=-xyz[:,:,1:];r=np.linalg.norm(xy,axis=2);theta=np.arctan2(r,xyz[:,:,0])
        radial=theta*(1+p[22]*theta**2+p[23]*theta**4)
        return p[19]*xy/np.maximum(r[:,:,None],1e-12)*radial[:,:,None]+p[20:22]

    def components(p,selection):
        frames,poses,local_corrected=model(p);uv=project(boundary,poses,p);mat_errors=[];global_errors=[]
        for i in np.flatnonzero(selection):
            x,y=uv[i].T
            weight=np.clip((x-9)/6,0,1)*np.clip((631-x)/6,0,1)*np.clip((y-9)/6,0,1)*np.clip((471-y)/6,0,1)
            weight*=~((x>155)&(x<505)&(y>270))
            mat_errors.extend(np.clip(map_coordinates(fields[i],[y,x],order=1,mode='nearest'),-50,50)*weight)
        # Local fixed and moving features constrain optical correction and jaw map.
        identity=np.eye(4)[None];identity[0]=local_corrected
        local_uv=project(fixed_features,identity,p)[0];fixed_error=(local_uv-fixed_targets).ravel()
        relative=np.linalg.inv(frames[:,5])@frames[:,6]
        moving_points=np.einsum('nij,j->ni',relative[:,:3,:3],moving)+relative[:,:3,3]
        moving_uv=project(moving_points[:,None,:],np.broadcast_to(local_corrected,(n,4,4)),p)[:,0]
        jaw_errors=[]
        for i in np.flatnonzero(selection):
            target=records[i]['moving_tip_px']
            if target is not None and 283<target[1]<305 and target[0]<475:
                jaw_errors.extend((moving_uv[i]-target).tolist())
        for link,vertices in surfaces:
            world=np.einsum('nij,vj->nvi',frames[:,:,:, :][:,link,:3,:3],vertices)+frames[:,link,None,:3,3]
            projected=project_world(world.reshape(-1,3),external).reshape(n,-1,2)
            for i in np.flatnonzero(selection):
                x,y=projected[i].T
                global_errors.extend(np.clip(map_coordinates(global_fields[i],[y,x],order=1,mode='nearest'),0,25))
        return np.array(mat_errors),fixed_error,np.array(jaw_errors),np.array(global_errors),uv

    def plate_residuals(p,selection):
        if plate_layouts is None:return np.zeros(0),np.zeros(0)
        layouts=plate_layouts.copy();layouts[:8]=p[26:].reshape(8,3)
        values=layouts[np.array([r['episode'] for r in records])]
        c,s=np.cos(values[:,2]),np.sin(values[:,2])
        rotation=np.zeros((n,2,2));rotation[:,0,0]=c;rotation[:,0,1]=-s;rotation[:,1,0]=s;rotation[:,1,1]=c
        xy=np.einsum('nij,vj->nvi',rotation,plate_boundary)+values[:,None,:2]
        points=np.concatenate([xy,np.full((n,len(plate_boundary),1),.003)],axis=2)
        _,poses,_=model(p);uv=project(points,poses,p);errors=[];external_errors=[]
        for i in np.flatnonzero(selection):
            if plate_fields[i] is None:continue
            x,y=uv[i].T
            weight=np.clip((x-8)/5,0,1)*np.clip((632-x)/5,0,1)*np.clip((y-8)/5,0,1)*np.clip((472-y)/5,0,1)
            weight*=~((x>150)&(x<510)&(y>270))
            weight*=1-map_coordinates(occlusion_fields[i],[y,x],order=1,mode='nearest')
            errors.extend(np.clip(map_coordinates(plate_fields[i],[y,x],order=1,mode='nearest'),-35,35)*weight)
        for ep in range(8):
            i=next(i for i,r in enumerate(records) if r['episode']==ep)
            projected=project_world(points[i],plate_external)
            external_errors.extend(map_coordinates(external_plate_fields[ep],[projected[:,1],projected[:,0]],order=1,mode='nearest'))
        return np.array(errors),np.array(external_errors)

    clearance_states=None;clearance_vertices=[]
    if a.floor_clearance:
        from scipy.spatial import ConvexHull
        trajectory=np.load(a.trajectory);all_training=trajectory['episode_index']<8
        # Convex-hull extrema preserve exact minimum height of the complete CAD
        # meshes; they are used only for calibration, not collision replacement.
        for name,link in [('fixed_jaw',5),('moving_jaw',6)]:
            vertices=np.array([[float(x) for x in line.split()[1:4]] for line in (ROOT/'assets/meshes'/f'{name}_collision.obj').read_text().splitlines() if line.startswith('v ')])
            clearance_vertices.append((link,vertices[ConvexHull(vertices).vertices]))
        source_states=trajectory['observation_state'][all_training]
        baseline_frames,_,_=model(start,source_states)
        heights=np.min([np.min(np.einsum('ni,vi->nv',baseline_frames[:,link,2,:3],vertices)+baseline_frames[:,link,2,3,None],axis=1) for link,vertices in clearance_vertices],axis=0)
        selection=heights<.025
        clearance_states=source_states[selection]
        print(f'Floor constraint: {len(clearance_states)} training states near the mat',flush=True)

    def floor_heights(p,source_states):
        frames,_,_=model(p,source_states)
        return np.array([np.min(np.einsum('ni,vi->nv',frames[:,link,2,:3],vertices)+frames[:,link,2,3,None],axis=1) for link,vertices in clearance_vertices])

    calls=[0]
    def residual(p):
        mat,fixed,jaw,glob,_=components(p,train)
        plate,external_plate=plate_residuals(p,train)
        # Per-frame mask edges are approximate; global cream is a coarse outline.
        priors=np.r_[(p[:5]-p0[:5])/.25,(p[5:10]-p0[5:10])/18,p[10:13]/[25,30,.1],
            p[13:16]/5,p[16:19]/4,(p[19]-300)/45,(p[20:22]-[320,240])/[22,25],
            (p[22:24]-p0[22:24])/[.12,.06],(p[24:26]-p0[24:26])/[.4,10]]
        if a.plate_constraints:
            priors=np.r_[priors,(p[:10]-start[:10])/np.r_[[.08]*5,[5]*5],
                (p[26:]-start[26:])/np.tile([.004,.004,.15],8)]
        if encoder_scales is not None:
            priors=np.r_[priors,(p[:5]-encoder_scales[:5])/.0001,(p[24]-encoder_scales[5])/.0002]
        if home_offsets is not None and a.home_prior_deg:
            # Correlated edge samples do not independently measure motor zeros.
            # The hand-set home is a soft prior, not an exact geometric datum.
            priors=np.r_[priors,(p[5:10]-home_offsets)/a.home_prior_deg*np.sqrt(train.sum())]
        calls[0]+=1
        if calls[0]%100==1:print(f'iteration evaluations {calls[0]}: mat {np.sqrt(np.mean(mat**2)):.2f}px, fixed {np.sqrt(np.mean(fixed**2)):.2f}px, jaw {np.sqrt(np.mean(jaw**2)):.2f}px, global {np.sqrt(np.mean(glob**2)):.2f}px',flush=True)
        floor=np.zeros(0) if clearance_states is None else np.maximum(0.,.0005-floor_heights(p,clearance_states)).ravel()/.0001
        # The image loss may reject an occluded edge; a rigid jaw cannot reject
        # the floor. Precondition this term so soft_l1 retains a quadratic
        # penalty for geometrically impossible penetration.
        floor=floor*np.sqrt(1+floor*floor)
        theta=np.linspace(0,1.8,24);derivative=1+3*p[22]*theta**2+5*p[23]*theta**4
        corners=np.array([[0.,0.],[640.,0.],[0.,480.],[640.,480.]])
        needed=np.max(np.linalg.norm((corners-p[20:22])/p[19],axis=1));extent=1.8*(1+p[22]*1.8**2+p[23]*1.8**4)
        optics=np.r_[np.maximum(0,.08-derivative),max(0,needed+.01-extent)]*1000
        optics=optics*np.sqrt(1+optics*optics)
        return np.r_[mat/5,fixed*300,jaw*.5,glob/9,plate/2.5,external_plate/1.5,priors,floor,optics]

    before={}
    for name,selection in [('train',train),('held_out',~train)]:
        m,f,j,g,_=components(p0,selection);before[name]=dict(mat_boundary_rms_px=float(np.sqrt(np.mean(m*m))),fixed_jaw_rms_px=float(np.sqrt(np.mean(f*f))),moving_tip_rms_px=float(np.sqrt(np.mean(j*j))),global_outline_rms_px=float(np.sqrt(np.mean(g*g))))
    start=np.clip(start,lower+1e-9,upper-1e-9)
    if encoder_scales is not None or a.fixed_wrist_lens:
        # Remove known scales from the optimization entirely. Tight numeric
        # bounds/large penalty columns otherwise spoil Jacobian conditioning.
        frozen=[]
        if encoder_scales is not None:
            start[:5]=encoder_scales[:5];start[24]=encoder_scales[5];frozen.extend([0,1,2,3,4,24])
        if a.fixed_wrist_lens:
            fixed=json.loads(a.fixed_wrist_lens.read_text())['lens']
            from reference_camera import validate_lens
            validate_lens(fixed)
            start[19:24]=[fixed['focal_px'],*fixed['principal_point_px'],*fixed['theta_coefficients']];frozen.extend(range(19,24))
        active=np.array([i for i in range(len(start)) if i not in frozen])
        def expand(values):
            p=start.copy();p[active]=values;return p
        fitted=least_squares(lambda values:residual(expand(values)),start[active],
            bounds=(lower[active],upper[active]),x_scale='jac',loss='soft_l1',f_scale=1.0,
            max_nfev=a.max_nfev,ftol=2e-6,diff_step=2e-4)
        p=expand(fitted.x)
    else:
        fitted=least_squares(residual,start,bounds=(lower,upper),x_scale='jac',loss='soft_l1',f_scale=1.0,max_nfev=a.max_nfev,ftol=2e-6,diff_step=2e-4)
        p=fitted.x
    frames,poses,new_local=model(p);after={}
    for name,selection in [('train',train),('held_out',~train)]:
        m,f,j,g,_=components(p,selection);after[name]=dict(mat_boundary_rms_px=float(np.sqrt(np.mean(m*m))),fixed_jaw_rms_px=float(np.sqrt(np.mean(f*f))),moving_tip_rms_px=float(np.sqrt(np.mean(j*j))),global_outline_rms_px=float(np.sqrt(np.mean(g*g))))
        plate,_=plate_residuals(p,selection)
        if len(plate):after[name]['plate_boundary_rms_px']=float(np.sqrt(np.mean(plate**2)))
    new_lens=dict(model='equidistant_polynomial',focal_px=float(p[19]),principal_point_px=p[20:22].tolist(),theta_coefficients=p[22:24].tolist())
    from reference_camera import validate_lens
    validate_lens(new_lens)
    report=dict(method='shared multi-episode metric reprojection fit with CAD and measured mat dimensions',
        training_episodes=list(range(8)),validation_episodes=[8,9],paired_views=n,before=before,after=after,
        scale=[*p[:5].tolist(),float(p[24])],offset_deg=[*p[5:10].tolist(),float(p[25])],
        base_translation_delta_m=(p[10:13]*.001).tolist(),wrist_link_from_optical=new_local.tolist(),wrist_lens=new_lens,
        parameters=p.tolist(),initial_parameters=p0.tolist(),solver=dict(success=bool(fitted.success),message=fitted.message,nfev=fitted.nfev),
        external_camera_constraint=external,
        wrist_lens_held_fixed=bool(a.fixed_wrist_lens),
        assumptions=['Metric scale comes from enclosure/mat and CAD dimensions',
            'No per-frame camera, robot or motor correction; last two episodes held out',
            'Visible cream outline and tip measurements have occlusion/segmentation uncertainty',
            'Image-derived calibration; hardware encoder calibration and measured contact properties remain unavailable'])
    if encoder_scales is not None:
        report['encoder_scale_constraint']=json.loads((a.encoder_scales or ROOT/'assets/encoder_units.json').read_text())
    if hardware is not None:
        report['motor_calibration']=hardware
        report['home_prior_std_deg']=a.home_prior_deg
        report['offset_correction_from_nominal_home_deg']=(p[5:10]-home_offsets).tolist()
        report['assumptions'][-1]='Hardware normalization verified against saved follower/leader calibrations; physical CAD zeros and contact properties still need geometric/dynamic identification'
    if clearance_states is not None:
        report['floor_clearance_constraint']=dict(training_states=len(clearance_states),target_minimum_z_m=.0005,
            before_minimum_z_m=float(floor_heights(start,clearance_states).min()),after_minimum_z_m=float(floor_heights(p,clearance_states).min()),
            method='Exact CAD extrema, training episodes only; one-sided floor clearance, no cube or endpoint fitting')
    (a.output/'fit.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(a.output/'fitted_poses.npz',indices=np.array([r['index'] for r in records]),poses=poses)
    print(json.dumps({k:report[k] for k in ['before','after','base_translation_delta_m','wrist_lens','scale','offset_deg','solver']},indent=2))
    # Review the measured and fitted edges across all held-out images.
    for i,row in enumerate(records):
        if row['training']:continue
        im=cv2.imread(str(a.output/f'{row["index"]:06d}_wrist.png'))
        predicted=project(boundary,poses,p)[i]
        for x,y in predicted:
            if 0<x<640 and 0<y<480:cv2.circle(im,(round(x),round(y)),1,(0,0,255),-1)
        cv2.imwrite(str(a.output/f'{row["index"]:06d}_validation.jpg'),im)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trajectory',type=Path,default=ROOT/'output/video_fit/reference_trajectory.npz')
    p.add_argument('--dataset',type=Path,default=ROOT.parent/'ref_data/green_cube_red_plate_1')
    p.add_argument('--output',type=Path,default=ROOT/'output/metric_refit/calibration')
    p.add_argument('--prepare-only',action='store_true');p.add_argument('--max-nfev',type=int,default=150)
    p.add_argument('--initial-fit',type=Path)
    p.add_argument('--plate-constraints',action='store_true')
    p.add_argument('--encoder-scales',type=Path)
    p.add_argument('--motor-calibration',type=Path)
    p.add_argument('--home-prior-deg',type=float,help='Soft uncertainty of the hand-set nominal CAD home, in degrees')
    p.add_argument('--floor-clearance',action='store_true',help='Constrain recorded CAD jaws above the mat; training episodes only')
    p.add_argument('--fixed-wrist-lens',type=Path,help='Retain an already validated wrist lens to reduce pose/lens ambiguity')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if not (a.output/'measurements.json').exists():prepare(a.trajectory,a.dataset,a.output)
    if not a.prepare_only:fit(a)


if __name__=='__main__':main()
