#!/usr/bin/env python3
"""Generate deterministic surface detail and the inflatable tube; no CAD tools.

Run from anywhere with NumPy and Pillow. Generated assets ship with the demo.
No real photograph is baked into these materials.
"""
from pathlib import Path
import sys
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scene_config import WIDTH,DEPTH,HEIGHT,SHEET_SIZE,SHEET_CENTER
from appearance import TUBE_LENGTH,TUBE_RADIUS,TUBE_NORMAL,TUBE_FRAME,TUBE_WHITE_COSINE,COLORS


def obj(path, vertices, faces, uv=None):
    vertices=np.array(vertices);faces=np.array(faces)
    normals=np.zeros_like(vertices)
    triangles=vertices[faces]
    fn=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
    for corner in range(3):np.add.at(normals,faces[:,corner],fn)
    normals/=np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-15)
    with path.open('w') as f:
        f.write('# Deterministic procedural lightbox detail, SI meters\n')
        for v in vertices:f.write('v '+' '.join(f'{x:.8f}' for x in v)+'\n')
        if uv is not None:
            for v in uv:f.write('vt '+' '.join(f'{x:.8f}' for x in v)+'\n')
        for v in normals:f.write('vn '+' '.join(f'{x:.8f}' for x in v)+'\n')
        for face in faces:
            f.write('f '+' '.join(f'{i+1}/{i+1}/{i+1}' if uv is not None else f'{i+1}//{i+1}' for i in face)+'\n')


def make_mat():
    w,d,t=SHEET_SIZE;nx,ny=121,85;radius=.007
    vertices=[];uv=[];faces=[]
    for j,y in enumerate(np.linspace(-d/2,d/2,ny)):
        dy=max(0.,abs(y)-(d/2-radius))
        xmax=w/2-radius+np.sqrt(max(0.,radius**2-dy**2))
        for i,x in enumerate(np.linspace(-xmax,xmax,nx)):
            # Shallow folds in the photographed sheet, below the collision tolerance.
            fade=np.sin(np.pi*i/(nx-1))*np.sin(np.pi*j/(ny-1))
            crease=.00032*np.exp(-((x-.014-.10*y)/.006)**2)
            crease+=.00040*np.exp(-((y+.018+.035*x)/.004)**2)
            ripple=.00006*(1+np.sin(29*x+7*y))*np.sin(43*y)**2
            vertices.append((x,y+SHEET_CENTER[1],t+.000015+fade*(crease+ripple)))
            uv.append((x/w+.5,y/d+.5))
            if i and j:
                a=j*nx+i;faces.extend([(a-nx-1,a-nx,a),(a-nx-1,a,a-1)])
    obj(ROOT/'assets/meshes/floor_sheet_detail.obj',vertices,faces,uv)


def make_tube():
    # Closed, softly tapered inflatable shell with a black backing. Separate
    # materials preserve the illuminated face and the light-absorbing back.
    vertices=[];groups={'diffuser':[],'black':[]};nx,nt=101,96
    local_normal=TUBE_FRAME.T@TUBE_NORMAL
    for i,x in enumerate(np.linspace(-TUBE_LENGTH/2,TUBE_LENGTH/2,nx)):
        end_distance=TUBE_LENGTH/2-abs(x)
        taper=np.sqrt(max(.004,1-(max(0.,TUBE_RADIUS-end_distance)/TUBE_RADIUS)**2))
        for j,theta in enumerate(np.linspace(0,2*np.pi,nt,endpoint=False)):
            wrinkle=1+.007*np.sin(23*x+3*theta)+.003*np.sin(69*x-5*theta)
            r=TUBE_RADIUS*taper*wrinkle
            vertices.append((x,r*np.cos(theta),r*np.sin(theta)))
            if i:
                a=i*nt+j;b=i*nt+(j+1)%nt
                radial=np.array([0,np.cos(theta+np.pi/nt),np.sin(theta+np.pi/nt)])
                group='diffuser' if radial@local_normal>TUBE_WHITE_COSINE else 'black'
                groups[group].extend([(a-nt,b-nt,b),(a-nt,b,a)])
    for group,faces in groups.items():obj(ROOT/f'assets/meshes/gc21b_{group}.obj',vertices,faces)


def make_panels():
    """Thin, slightly bowed boards with actual uneven gaps above the floor."""
    def grid(name,nu,nv,position):
        vertices=[];uv=[];faces=[]
        for j,v in enumerate(np.linspace(0,1,nv)):
            for i,u in enumerate(np.linspace(0,1,nu)):
                vertices.append(position(u,v));uv.append((u,v))
                if i and j:
                    a=j*nu+i;faces.extend([(a-nu-1,a-nu,a),(a-nu-1,a,a-1)])
        if name in ['right','ceiling']:faces=[(a,c,b) for a,b,c in faces]
        obj(ROOT/f'assets/meshes/panel_{name}.obj',vertices,faces,uv)
    def floor(u,v):
        x=(u-.5)*WIDTH;y=v*DEPTH
        edge=np.exp(-v*70)+np.exp(-(1-v)*70)
        curl=.0014*edge*(.45+.55*np.sin(u*np.pi)**2)
        return x,y,curl
    grid('floor',121,81,floor)
    for side in [-1,1]:
        def wall(u,v,side=side):
            y=u*DEPTH
            gap=.0018+.0012*np.sin(3.2*np.pi*u+.3)**2
            x=side*(WIDTH/2+.0011*np.sin(np.pi*u)*np.sin(1.3*np.pi*v))
            return x,y,gap+v*(HEIGHT-gap)
        grid('left' if side<0 else 'right',91,61,wall)
    def rear(u,v):
        gap=.0016+.0012*np.sin(4.1*np.pi*u)**2
        return (u-.5)*WIDTH,DEPTH+.0013*np.sin(np.pi*u)*np.sin(1.5*np.pi*v),gap+v*(HEIGHT-gap)
    grid('rear',121,61,rear)
    grid('ceiling',81,61,lambda u,v:((u-.5)*WIDTH,v*DEPTH,HEIGHT+.001*np.sin(np.pi*u)*np.sin(2*np.pi*v)))


def make_adapter_screw():
    """Domed polished head with a recessed Phillips opening, local axis +X."""
    radius=.0024;count=101;vertices=[];faces=[];uv=[]
    values=np.linspace(-radius,radius,count)
    for j,z in enumerate(values):
        for i,y in enumerate(values):
            rr=(y*y+z*z)/radius**2
            x=.00070-.00028*min(rr,1)
            vertices.append((x,y,z));uv.append((i/(count-1),j/(count-1)))
            if not i or not j:continue
            corners=[(values[i-1],values[j-1]),(y,values[j-1]),(y,z),(values[i-1],z)]
            def solid(a,b):
                cross=(abs(a)<.00028 and abs(b)<.00155) or (abs(b)<.00028 and abs(a)<.00155)
                return a*a+b*b<radius**2 and not cross
            if all(solid(a,b) for a,b in corners):
                a=j*count+i;faces.extend([(a-count-1,a-count,a),(a-count-1,a,a-1)])
    # Curved outer bevel/sidewall catches a narrow tube highlight.
    first=len(vertices);nt=96
    for x,r in [(-.00035,.00225),(.00015,.0024),(.00042,.0024)]:
        for theta in np.linspace(0,2*np.pi,nt,endpoint=False):
            vertices.append((x,r*np.cos(theta),r*np.sin(theta)));uv.append((0,0))
    for row in range(2):
        for j in range(nt):
            a=first+row*nt+j;b=first+row*nt+(j+1)%nt
            faces.extend([(a,b,b+nt),(a,b+nt,a+nt)])
    obj(ROOT/'assets/meshes/adapter_screw_head.obj',vertices,faces,uv)


def make_textures():
    target=ROOT/'assets/textures';target.mkdir(exist_ok=True)
    rng=np.random.default_rng(20261019)
    for name,size,noise in [('floor_sheet',(1536,1024),.070),('white',(768,768),.004)]:
        w,h=size;x,y=np.meshgrid(np.arange(w),np.arange(h))
        grain=1+rng.normal(0,noise,(h,w))
        if name=='floor_sheet':
            grain+=.009*np.cos(x*np.pi)*np.sin(y*np.pi/2)
            grain+=.018*np.sin(x*.027+y*.019)*np.sin(y*.041)
        linear=np.clip(grain[:,:,None]*np.array(COLORS[name]),0,1)
        # Chrono decodes diffuse textures from sRGB with power 2.2.
        Image.fromarray(np.uint8(np.clip(linear**(1/2.2)*255,0,255))).save(target/f'{name}.png')


if __name__=='__main__':
    make_mat();make_tube();make_panels();make_adapter_screw();make_textures()
    print('Generated bowed panels/seams, textured mat, GC21B shell and domed adapter screws.')
