"""CAD-frame SO-101 forward/inverse kinematics (NumPy only).

Joint coordinates are radians internally, offset from the supplied STEP pose.
These are simulation coordinates, not calibrated LeRobot motor coordinates.
"""
from dataclasses import dataclass
import math
import numpy as np

JOINT_NAMES=('shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll','gripper')
# Compact left-side rest: elbow folded back, wrist/gripper pointing down.
# This also clears the overhead camera's spawn-area sightlines.
HOME_DEG=np.array([90.,35.,-50.,105.,90.,58.])
START_POSES_DEG={
    'folded':HOME_DEG,
    'side':np.array([90.,105.,-70.,-5.,90.,58.]),
}
OPEN_DEG=58.
GRASP_DEG=84.  # servo target; physical contacts limit the actual closing angle
TCP_POINT=np.array([.007,-.210,.156])
APPROACH=np.array([0.,-.9999619,-.0087265])
CLOSING=np.array([-.0174524,0.,.9998477])
# Center the 30 mm cube clear of the fixed tooth tips during vertical descent.
# The former target overlapped the fixed jaw by approximately 2.4 mm.
TCP_POINT=TCP_POINT+.0035*CLOSING
LIMITS_DEG=np.array([[-170,170],[-20,190],[-170,60],[-150,150],[-180,180],[45,108]],dtype=float)

def unit(v):
    v=np.asarray(v,dtype=float)
    return v/np.linalg.norm(v)

def axis_rotation(axis,angle):
    x,y,z=unit(axis)
    k=np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])
    return np.eye(3)+math.sin(angle)*k+(1-math.cos(angle))*(k@k)

def transform(r=None,p=None):
    t=np.eye(4)
    if r is not None:t[:3,:3]=r
    if p is not None:t[:3,3]=p
    return t

def forward_kinematics(assets,angles_deg):
    t=transform(np.diag([-1.,-1.,1.]),[0,.0540,-.00039])
    frames=[t.copy()]
    for joint,angle in zip(assets['joints'],angles_deg):
        r=axis_rotation(joint['axis'],math.radians(angle));p=np.asarray(joint['pivot_m'])
        t=t@transform(r,p-r@p);frames.append(t.copy())
    return frames

def tool_pose(assets,q_rad):
    f=forward_kinematics(assets,np.rad2deg(q_rad))[5]
    approach=unit(f[:3,:3]@APPROACH)
    closing=unit(f[:3,:3]@CLOSING)
    closing=unit(closing-approach*np.dot(approach,closing))
    return transform(np.column_stack((approach,closing,np.cross(approach,closing))),f[:3,:3]@TCP_POINT+f[:3,3])

@dataclass
class IKResult:
    q: np.ndarray
    success: bool
    position_error_m: float
    approach_error_deg: float
    closing_error_deg: float

class SO101IK:
    def __init__(self,assets):
        self.assets=assets
        self.limits=np.deg2rad(LIMITS_DEG)

    def solve(self,position,seed=None,yaw=0.,jaw_deg=GRASP_DEG):
        target=np.asarray(position,dtype=float)
        if target.shape!=(3,) or not np.isfinite(target).all():raise ValueError('IK target must contain 3 finite meters')
        closing=np.array([math.cos(yaw),math.sin(yaw),0.])
        def residual(x):
            q=np.r_[x,math.radians(jaw_deg)]
            t=tool_pose(self.assets,q)
            return np.r_[(t[:3,3]-target)*40.,(t[:3,0]-[0,0,-1]),(t[:3,1]-closing)*.5]
        starts=[np.deg2rad(HOME_DEG) if seed is None else np.asarray(seed,dtype=float)]
        # Alternate elbow/pitch branches are tried only if the warm start fails.
        starts.extend(np.deg2rad([0,lift,elbow,90-lift-elbow,90,jaw_deg]) for lift,elbow in [(110,-90),(140,-120),(75,-50)])
        best=None;best_cost=float('inf')
        for start in starts:
            x=np.clip(start[:5],self.limits[:5,0],self.limits[:5,1]).copy()
            damping=1e-3
            for iteration in range(100):
                r=residual(x);cost=float(r@r)
                if cost<best_cost:best_cost=cost;best=x.copy()
                jac=np.column_stack([(residual(x+np.eye(5)[j]*1e-4)-r)/1e-4 for j in range(5)])
                delta=-np.linalg.solve(jac.T@jac+damping*np.eye(5),jac.T@r)
                delta*=min(1.,.25/max(np.linalg.norm(delta),1e-12))
                accepted=False
                for scale in [1.,.5,.25,.1]:
                    trial=np.clip(x+scale*delta,self.limits[:5,0],self.limits[:5,1])
                    rr=residual(trial)
                    if rr@rr<cost:
                        x=trial;damping=max(1e-6,damping*.6);accepted=True;break
                if not accepted:damping*=10
                if np.linalg.norm(delta)<1e-7 or damping>1e7:break
            result=self.result(np.r_[best,math.radians(jaw_deg)],target,closing)
            if result.success:return result
        return self.result(np.r_[best,math.radians(jaw_deg)],target,closing)

    def result(self,q,target,closing):
        t=tool_pose(self.assets,q)
        pos=float(np.linalg.norm(t[:3,3]-target))
        angle=math.degrees(math.acos(float(np.clip(-t[2,0],-1,1))))
        close=math.degrees(math.acos(float(np.clip(np.dot(t[:3,1],closing),-1,1))))
        return IKResult(q.copy(),pos<.0015 and angle<4. and close<6.,pos,angle,close)
