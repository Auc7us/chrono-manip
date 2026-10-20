#!/usr/bin/env python3
"""Fit a finite capsule diffuser's silhouette/backing to the fixed-camera photo.

Uses known 90 cm nominal length, bounded diameter/ceiling position and the fitted
camera. The image is a measurement target; it never becomes a scene texture.
"""
from pathlib import Path
import argparse
import json
import sys
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.special import expit

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from reference_camera import PHOTO_FIT_PROFILE,load_profile
from scene_config import WIDTH,DEPTH,HEIGHT


def rays(profile,x,y):
    lens=profile['lens'];f=lens['focal_px'];cx,cy=lens['principal_point_px'];k1,k2=lens['theta_coefficients']
    dx=(x-cx)/f;dy=(y-cy)/f;rd=np.sqrt(dx*dx+dy*dy);theta=rd.copy()
    for _ in range(10):
        t2=theta*theta;theta-=(theta*(1+k1*t2+k2*t2*t2)-rd)/(1+3*k1*t2+5*k2*t2*t2)
    scale=np.sin(theta)/np.maximum(rd,1e-10)
    local=np.column_stack([np.cos(theta),-scale*dx,-scale*dy])
    return local@np.array(profile['world_from_optical_rotation']).T


def capsule(p,directions,origin):
    center=p[:3];yaw,pitch=np.deg2rad(p[3:5]);length,radius=p[5:7]
    axis=np.array([np.cos(yaw)*np.cos(pitch),np.sin(yaw)*np.cos(pitch),np.sin(pitch)])
    a=center-axis*(length/2-radius);b=center+axis*(length/2-radius);ba=b-a;oa=origin-a
    baba=ba@ba;bard=directions@ba;baoa=ba@oa;rdoa=directions@oa;oaoa=oa@oa
    denom=np.maximum(baba-bard*bard,1e-9)
    seg=np.clip((baoa-bard*rdoa)/denom,0,1)
    closest=a+seg[:,None]*ba
    t=np.sum((closest-origin)*directions,axis=1)
    distance=np.linalg.norm(origin+t[:,None]*directions-closest,axis=1)
    # Smooth boundary with about 1.5-pixel transition for optimizer gradients.
    occupancy=expit((radius-distance)/np.maximum(t/302*1.5,.0004))*(t>0)
    qb=baba*rdoa-baoa*bard;qc=baba*oaoa-baoa*baoa-radius*radius*baba
    discriminant=qb*qb-denom*qc
    tc=(-qb-np.sqrt(np.maximum(discriminant,0)))/denom
    along=baoa+tc*bard
    tc=np.where((discriminant>0)&(along>0)&(along<baba)&(tc>0),tc,np.inf)
    all_t=[tc]
    for cap in [a,b]:
        oc=origin-cap;dot=directions@oc;disc=dot*dot-oc@oc+radius*radius
        ts=-dot-np.sqrt(np.maximum(disc,0));all_t.append(np.where((disc>0)&(ts>0),ts,np.inf))
    hit_t=np.min(all_t,axis=0)
    hit=origin+np.where(np.isfinite(hit_t),hit_t,t)[:,None]*directions
    seg=np.clip(((hit-a)@ba)/baba,0,1);normal=hit-(a+seg[:,None]*ba)
    normal/=np.maximum(np.linalg.norm(normal,axis=1,keepdims=True),1e-9)
    angle=np.deg2rad(p[7]);emission=np.array([0,np.sin(angle),-np.cos(angle)])
    emission-=axis*(axis@emission);emission/=np.linalg.norm(emission)
    white=expit((normal@emission-p[8])/.045)
    return occupancy,occupancy*white,axis,emission


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('photo',type=Path)
    parser.add_argument('--output',type=Path,default=ROOT/'assets/reference_tube.json')
    args=parser.parse_args();profile=load_profile(PHOTO_FIT_PROFILE)
    image=cv2.imread(str(args.photo));gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    y,x=np.mgrid[0:465:3,320:640:3];px=x.ravel().astype(float)+.5;py=y.ravel().astype(float)+.5
    value=gray[y,x].ravel();black=(value<90).astype(float);white=(value>212).astype(float)
    # Only this upper-right ROI contains the tube. Remove tiny ceiling glints.
    keep=(px>350)|(py<40);black*=keep;white*=keep
    target=np.r_[black,white]
    directions=rays(profile,px,py);origin=np.array(profile['optical_position_m'])
    initial=[.025,DEPTH-.04,HEIGHT-.05,0,0,.90,.052,-15,.35]
    lower=[-.12,DEPTH-.13,HEIGHT-.13,-10,-12,.86,.025,-100,-.4]
    upper=[ .16,DEPTH+.03,HEIGHT-.015,10,12,.94,.090,50,.85]
    def residual(p):
        occ,wh,_,_=capsule(p,directions,origin)
        return np.r_[occ-wh-black,wh-white,(p[5]-.90)/.05,(p[3]/15),(p[4]/15)]
    fit=least_squares(residual,initial,bounds=(lower,upper),diff_step=1e-4,max_nfev=350,
                      x_scale='jac',ftol=1e-7)
    p=fit.x;occ,wh,axis,normal=capsule(p,directions,origin)
    score=lambda a,b:float(np.sum((a>.5)&(b>.5))/max(1,np.sum((a>.5)|(b>.5))))
    report=dict(source=args.photo.name,method='bounded capsule/backing silhouette fit to fixed-camera image',
        center_m=p[:3].tolist(),axis=axis.tolist(),emission_normal=normal.tolist(),
        installed_length_m=float(p[5]),radius_m=float(p[6]),white_sector_cosine=float(p[8]),
        black_mask_iou=score(occ-wh,black),white_mask_iou=score(wh,white),
        assumptions=['90 cm nominal length prior','Circular section and rounded capsule ends',
                     'Backing segmentation from upper-right image region'])
    args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
