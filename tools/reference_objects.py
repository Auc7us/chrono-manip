"""Fit initial cube/plate geometry to their two measured silhouettes."""
import itertools
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt,map_coordinates
from scipy.optimize import least_squares
from reference_camera import project_world,pixel_rays


def object_mask(image,green=True):
    f=image.astype(float);r,g,b=f[:,:,0],f[:,:,1],f[:,:,2]
    # Dark cube faces remain part of its silhouette. A threshold of 45 excluded
    # much of the shaded front face and biased its measured center upward.
    mask=((g>r*1.5)&(g>b*1.25)&(g>20)) if green else ((r>g*1.7)&(r>b*1.5)&(r>60))
    n,labels,stats,centers=cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    candidates=[i for i in range(1,n) if stats[i,cv2.CC_STAT_AREA]>(12 if green else 20)]
    if not candidates:return None
    i=max(candidates,key=lambda i:stats[i,cv2.CC_STAT_AREA])
    return (labels==i).astype(np.uint8),centers[i]


def box_vertices(center,yaw,size):
    local=np.array(list(itertools.product([-1.,1.],repeat=3)))*np.asarray(size)/2
    c,s=np.cos(yaw),np.sin(yaw);r=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
    return local@r.T+center


def hull_boundary(vertices,profile,n=12):
    uv=project_world(vertices,profile)
    hull=cv2.convexHull(uv.astype(np.float32)).reshape(-1,2)
    return np.concatenate([a+(b-a)*np.linspace(0,1,n)[:,None] for a,b in zip(hull,np.roll(hull,-1,axis=0))]),hull


def fit_initial_objects(images,profiles,cube_center_z=.016):
    result={}
    for name,green,size,z in [('cube',True,[.03]*3,cube_center_z),('plate',False,[.05,.05,.002],.002)]:
        observations={}
        for cam,image in images.items():
            item=object_mask(image,green)
            if item is None:continue
            mask,center=item
            contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
            contour=max(contours,key=len).reshape(-1,2).astype(float)
            contour=contour[::max(1,len(contour)//90)]
            field=distance_transform_edt(mask==0)-distance_transform_edt(mask!=0)
            observations[cam]=(mask,center,contour,field)
        if not observations:raise ValueError(f'No {name} found in either initial camera frame')
        cam='external' if 'external' in observations else 'wrist'
        p=profiles[cam];o=np.array(p['optical_position_m']);d=pixel_rays([observations[cam][1]],p)[0]
        initial=o+d*((z-o[2])/d[2]);initial[2]=z
        def residual(v):
            vertices=box_vertices([*v[:2],z],v[2],size);errors=[]
            for camera,(mask,center,contour,field) in observations.items():
                predicted,hull=hull_boundary(vertices,profiles[camera])
                # Fixed-length edge samples keep the residual dimension stable
                # when the visible convex-hull vertex count changes.
                u=np.linspace(0,len(predicted)-1,96)
                prediction=np.column_stack([np.interp(u,np.arange(len(predicted)),predicted[:,j]) for j in [0,1]])
                error=map_coordinates(field,[prediction[:,1],prediction[:,0]],order=1,mode='nearest')
                a=hull;b=np.roll(hull,-1,axis=0);ab=b-a
                t=np.clip(np.sum((contour[:,None,:]-a)*ab,axis=2)/np.maximum(np.sum(ab*ab,axis=1),1e-8),0,1)
                nearest=a+t[:,:,None]*ab
                distance=np.min(np.linalg.norm(contour[:,None,:]-nearest,axis=2),axis=1)
                weight=1. if camera=='wrist' else .65
                errors.extend(error*weight);errors.extend(distance*weight)
            return np.r_[errors,(v[:2]-initial[:2])/.04]
        best=None
        for yaw in [0.,np.pi/8,np.pi/4,-np.pi/8]:
            fit=least_squares(residual,[*initial[:2],yaw],bounds=([initial[0]-.06,initial[1]-.06,-np.pi/2],[initial[0]+.06,initial[1]+.06,np.pi/2]),
                diff_step=1e-3,loss='soft_l1',f_scale=1.5,max_nfev=100)
            if best is None or np.mean(fit.fun**2)<np.mean(best.fun**2):best=fit
        center=[*best.x[:2].tolist(),z]
        vertices=box_vertices(center,best.x[2],size)
        result[name]=dict(center_m=center,yaw_rad=float(best.x[2]),dimensions_m=size,
            reprojection_rms_px=float(np.sqrt(np.mean(best.fun[:-2]**2))),observations={})
        for camera,(mask,_,contour,field) in observations.items():
            _,hull=hull_boundary(vertices,profiles[camera]);pred=np.zeros_like(mask);cv2.fillConvexPoly(pred,np.rint(hull).astype(np.int32),1)
            iou=np.count_nonzero(pred&mask)/max(1,np.count_nonzero(pred|mask))
            result[name]['observations'][camera]=dict(silhouette_iou=float(iou),reference_area_px=int(mask.sum()))
    return result
