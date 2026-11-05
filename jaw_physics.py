"""Full CAD jaw collisions and a torque-limited physical gripper servo."""
import json
import math
from pathlib import Path
import numpy as np

GRASP_MODE='physical_jaw_contacts'
JAW_MASS=.025
JAW_KP=1.
JAW_KD=.015
# LeRobot's SO follower caps the gripper STS3215 at 50% torque and 25% when
# overloaded; 25% of its ~1.9 Nm stall torque bounds a sustained grip.
JAW_TORQUE_LIMIT=.45
CONTACT_FRICTION=.5

class JawPhysics:
    def __init__(self,scene,dynamic):
        self.scene=scene;self.dynamic=dynamic;self.torque=0.;self.contacts={}
        c=scene.c;root=Path(__file__).resolve().parent
        response_file=root/'assets/recorded_jaw_servo.json'
        self.response=getattr(scene.args,'jaw_response_override',None) or (json.loads(response_file.read_text()) if response_file.is_file() else None)
        self.command_history=[];self.filtered_target_deg=None;self.last_drive_time=None
        self.geometry=json.loads((root/'assets/jaw_collision.json').read_text())
        material=c.ChContactMaterialNSC();material.SetFriction(CONTACT_FRICTION);material.SetRestitution(0.)
        contact_file=root/'assets/recorded_contact_response.json'
        self.contact_response=json.loads(contact_file.read_text()) if contact_file.exists() else {}
        requested=getattr(scene.args,'contact_compliance_m_n',None)
        self.contact_compliance_m_n=float(self.contact_response.get('normal_compliance_m_per_n',0.) if requested is None else requested)
        self.contact_tangential_ratio=float(self.contact_response.get('tangential_compliance_ratio',.25))
        self.contact_damping_s=float(getattr(scene.args,'contact_damping_s',self.contact_response.get('damping_time_s',.002)))
        if self.contact_compliance_m_n<0:raise ValueError('Contact compliance must be nonnegative')
        if self.contact_compliance_m_n:
            material.SetCompliance(self.contact_compliance_m_n)
            material.SetComplianceT(self.contact_compliance_m_n*self.contact_tangential_ratio)
            material.SetDampingF(self.contact_damping_s)
        identifiers={}
        for name,part in self.geometry['parts'].items():
            body=scene.links[part['link']]
            mesh=c.ChTriangleMeshConnected.CreateFromWavefrontFile(str(root/'assets/meshes'/part['file']),True,False)
            body.AddCollisionShape(c.ChCollisionShapeTriangleMesh(material,mesh,False,False,0.))
            body.EnableCollision(True);body.GetCollisionModel().SetFamily(1)
            # Adjacent jaw CAD includes the common servo pivot. Contacts with
            # cube/floor/tray remain enabled; mechanism self-contact is excluded.
            body.GetCollisionModel().DisallowCollisionsWith(1)
            identifiers[body.GetIdentifier()]=name
        cube_id=scene.cube.GetIdentifier()
        environment_ids={scene.tray.GetIdentifier(),scene.world.GetIdentifier()}
        self.motor=None
        if dynamic:
            fixed=scene.links[5];part=self.geometry['parts']['fixed_jaw']
            fixed.SetMass(.08)
            x,y,z=part['dimensions_m']
            fixed.SetInertiaXX(scene.vec(.08*np.array([y*y+z*z,x*x+z*z,x*x+y*y])/12))
            fixed.SetFixed(False)
            # A moving contact body must stay active in the solver: teleporting
            # a fixed body loses its tangential contact velocity in NSC friction.
            self.motion=c.ChLinkMotionImposed()
            self.motion.Initialize(fixed,scene.world,False,scene.frame(scene.frames[5]),c.ChFramed())
            self.position=c.ChFunctionPositionSetpoint();self.rotation=c.ChFunctionRotationSetpoint()
            self.motion.SetPositionFunction(self.position);self.motion.SetRotationFunction(self.rotation)
            self.set_parent_pose(scene.frames[5],0.,reset=True)
            scene.system.AddLink(self.motion)
            jaw=scene.links[6];part=self.geometry['parts']['moving_jaw']
            jaw.SetMass(JAW_MASS)
            x,y,z=part['dimensions_m']
            jaw.SetInertiaXX(scene.vec(JAW_MASS*np.array([y*y+z*z,x*x+z*z,x*x+y*y])/12))
            jaw.SetFixed(False)
            joint=scene.assets['joints'][5];parent=scene.frames[5]
            pivot=parent[:3,:3]@np.array(joint['pivot_m'])+parent[:3,3]
            axis=parent[:3,:3]@np.array(joint['axis']);axis/=np.linalg.norm(axis)
            guide=np.array([0.,1.,0.]);x_axis=np.cross(guide,axis);x_axis/=np.linalg.norm(x_axis)
            rotation=np.column_stack((x_axis,np.cross(axis,x_axis),axis))
            from so101_ik import transform
            self.motor=c.ChLinkMotorRotationTorque()
            self.motor.Initialize(jaw,scene.links[5],scene.frame(transform(rotation,pivot)))
            self.function=c.ChFunctionConst(0.);self.motor.SetTorqueFunction(self.function)
            scene.system.AddLink(self.motor)

        class Reporter(c.ReportContactCallback):
            def __init__(self):super().__init__();self.pairs={};self.environment_penetration=0.
            def OnReportContact(self,pA,pB,plane,distance,radius,force,torque,modA,modB,offset):
                a=c.CastToChBody(modA);b=c.CastToChBody(modB)
                if a is None or b is None:return True
                ids={a.GetIdentifier(),b.GetIdentifier()}
                for identifier,name in identifiers.items():
                    if cube_id in ids and identifier in ids:
                        item=self.pairs.setdefault(name,dict(count=0,normal_force_n=0.,penetration_m=0.))
                        item['count']+=1;item['normal_force_n']+=abs(float(force.x))
                        item['penetration_m']=max(item['penetration_m'],max(0.,-float(distance)))
                    if identifier in ids and ids&environment_ids:
                        self.environment_penetration=max(self.environment_penetration,max(0.,-float(distance)))
                return True
        self.reporter=Reporter()

    def set_parent_pose(self,pose,t,reset=False):
        if not self.dynamic:return
        if reset:
            self.position.Reset(t);self.rotation.Reset(t)
            self.command_history=[];self.filtered_target_deg=None;self.last_drive_time=None
        self.position.SetSetpoint(self.scene.vec(pose[:3,3]),t)
        self.rotation.SetSetpoint(self.scene.quaternion(pose[:3,:3]),t)

    def angle_deg(self):
        scene=self.scene
        fixed=scene.body_transform(scene.links[5]);moving=scene.body_transform(scene.links[6])
        r=fixed[:3,:3].T@moving[:3,:3];axis=np.array(scene.assets['joints'][5]['axis']);axis/=np.linalg.norm(axis)
        sine=np.dot(axis,[r[2,1]-r[1,2],r[0,2]-r[2,0],r[1,0]-r[0,1]])/2
        return math.degrees(math.atan2(sine,(np.trace(r)-1)/2))

    def drive(self,target_deg,actual_deg):
        if not self.dynamic:return
        if self.response:
            t=float(self.scene.system.GetChTime())
            self.command_history.append((t,float(target_deg)))
            requested=t-self.response['command_latency_s']
            while len(self.command_history)>2 and self.command_history[1][0]<requested:
                self.command_history.pop(0)
            times,targets=zip(*self.command_history)
            delayed=float(np.interp(requested,times,targets))
            if self.filtered_target_deg is None:self.filtered_target_deg=float(actual_deg)
            elapsed=0. if self.last_drive_time is None else max(0.,t-self.last_drive_time)
            direction='close' if delayed>=self.filtered_target_deg else 'open'
            # Preserve the existing physical torque controller. Shape its target
            # to the measured response, accounting for its own KD/KP time scale.
            tau=max(.005,self.response[direction+'_time_constant_s']-JAW_KD/JAW_KP)
            self.filtered_target_deg+=(delayed-self.filtered_target_deg)*(-math.expm1(-elapsed/tau))
            self.last_drive_time=t;target_deg=self.filtered_target_deg
        error=math.radians(target_deg-actual_deg)
        self.torque=float(np.clip(JAW_KP*error-JAW_KD*self.motor.GetMotorAngleDt(),-JAW_TORQUE_LIMIT,JAW_TORQUE_LIMIT))
        self.function.SetConstant(self.torque)

    def observe(self):
        self.reporter.pairs={};self.reporter.environment_penetration=0.
        self.scene.system.GetContactContainer().ReportAllContacts(self.reporter)
        self.contacts=self.reporter.pairs
        return self.contacts
