"""Prepare portable OBJ assets; only regeneration needs numpy, scipy, trimesh, OCP.

Run inspect_cad.py with --mesh for the main assembly and gripper first.
All source STEP dimensions and feature measurements below are millimeters.
"""
from pathlib import Path
import json
import hashlib
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from inspect_cad import tessellate
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
from OCP.gp import gp_Pnt, gp_Dir, gp_Ax2

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
OUT = ASSETS / "meshes"

def rigid_fit(a, b):
    ca, cb = a.mean(0), b.mean(0)
    u, _, vt = np.linalg.svd((a-ca).T @ (b-cb))
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1] *= -1
        r = vt.T @ u.T
    return r, cb-r@ca

def write_obj(path, vertices, faces):
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    normals = mesh.vertex_normals
    with path.open("w") as f:
        f.write("# attempt-codex: meters, original assembly reference frame\n")
        for p in vertices:
            f.write("v %.8f %.8f %.8f\n" % tuple(p))
        for p in normals:
            f.write("vn %.6f %.6f %.6f\n" % tuple(p))
        for a,b,c in faces + 1:
            f.write(f"f {a}//{a} {b}//{b} {c}//{c}\n")

def plate_with_holes(width, height, thickness, holes):
    # Camera coordinate convention: optical axis +X; left +Y; up +Z.
    s = BRepPrimAPI_MakeBox(gp_Pnt(-thickness,-width/2,-height/2), thickness,width,height).Shape()
    for y,z,r in holes:
        tool = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(-thickness-1,y,z),gp_Dir(1,0,0)),r,thickness+2).Shape()
        s = BRepAlgoAPI_Cut(s,tool).Shape()
    return tessellate(s)

def main():
    OUT.mkdir(exist_ok=True)
    raw = np.load(ASSETS / "SO101_Assembly_raw.npz")
    report = json.loads((ASSETS / "SO101_Assembly_parts.json").read_text())
    # Four matching wrist-roll horn bores identify the gripper reference frame.
    source = np.array([[42.342,28.110,-34.922],[42.254,28.110,-25.023],
                       [42.342,38.009,-34.922],[42.254,38.009,-25.023]])
    target = np.array([[12.097,-127.084,147.146],[11.924,-127.171,157.044],
                       [2.199,-127.083,146.973],[2.026,-127.170,156.871]])
    ref_r, ref_t = rigid_fit(source,target)
    # STL and reference STEP are the same dovetail mount, with different camera
    # plates. Use the TWO dovetail screw axes, not the changed camera plate.
    stl_r = Rotation.from_euler("y",-.5,degrees=True).as_matrix()
    stl_t = np.array([23.3645,5.213,-30.8775]) - stl_r @ np.array([.520011,-28.218214,24.590169])
    mount_r, mount_t = ref_r @ stl_r, ref_r @ stl_t + ref_t
    # The rigid and soft gripper revisions differ by ~0.6 mm at the dovetail.
    # Correct across the screw axes using the ACTUAL rigid gripper's two bores.
    slider_bores=np.array([[.520011,-28.218214,28.640169],[.520011,-28.218214,20.540169]])
    rigid_bores=np.array([[27.505,-145.557,155.304],[27.646,-145.486,147.205]])
    screw_axis=mount_r@np.array([0,-1.,0])
    correction=(rigid_bores-(slider_bores@mount_r.T+mount_t)).mean(0)
    correction-=screw_axis*np.dot(correction,screw_axis)
    mount_t+=correction
    bore_errors=rigid_bores-(slider_bores@mount_r.T+mount_t)
    bore_errors-=np.outer(bore_errors@screw_axis,screw_axis)
    groups = {}
    primary_groups = {0:0,1:1,2:2,3:3,4:0,5:1,6:3,7:4,8:5,9:6,257:0}
    primary = [i for i in primary_groups if i !=257]
    primary_points = {i:raw[f"v{i}"][::max(1,len(raw[f"v{i}"])//2000)] for i in primary}
    from scipy.spatial import cKDTree
    trees = {i:cKDTree(v) for i,v in primary_points.items()}
    def add(group, material, v, f):
        groups.setdefault((group,material),[]).append((np.asarray(v)*.001,np.asarray(f,dtype=int)))
    for part in report:
        i = part["index"]
        v,f = raw[f"v{i}"],raw[f"f{i}"]
        if not len(v):
            continue
        # Replace the original single-color cable solids with three visible strands.
        if 251 <= i <=255 or i==258:
            continue
        if 204<=i<=250:
            starts=[204,212,220,228,236,243]
            motor = max(j for j,start in enumerate(starts) if start<=i)
            slot = i-starts[motor]
            # Interior gears, PCB, and DC motor are hidden by the intact housing.
            if slot in (3,4,5):
                continue
            group = motor if slot<6 else min(motor+1,6)
            color = "black" if slot<6 else "metal"
        elif i in primary_groups:
            group,color=primary_groups[i],"cream"
        else:
            center=(v.min(0)+v.max(0))/2
            closest=min(primary,key=lambda j:trees[j].query(center)[0])
            group=primary_groups[closest]
            color="metal" if i<100 or 192<=i<=201 or i==256 else "electronics"
            if "Wiring_holder" in "/".join(part["path"]):
                color="cream"
        add(group,color,v,f)
    manifest=[]
    for (group,color),pieces in sorted(groups.items()):
        vertices=[]; faces=[]; n=0
        for v,f in pieces:
            vertices.append(v); faces.append(f+n); n+=len(v)
        name=f"link_{group}_{color}.obj"
        write_obj(OUT/name,np.concatenate(vertices),np.concatenate(faces))
        manifest.append(dict(file=name,link=group,material=color))
    mount=trimesh.load(ASSETS/"cad/IMXmountSO101.stl")
    write_obj(OUT/"wrist_camera_mount.obj",(mount.vertices@mount_r.T+mount_t)*.001,mount.faces)
    # The STL has four circular slots, 21 mm across Z and 13.5 mm along its plate.
    # Fit the arc points to obtain actual centers rather than bounding-box centers.
    n=np.array([-.906307787,.422618262,0.])
    up=np.array([.422618262,.906307787,0.])
    plane_vertices=mount.vertices[np.unique(mount.facets_boundary[0])]
    uv=np.column_stack((plane_vertices@up,plane_vertices[:,2]))
    centers=[]
    for a in [-60.35,-46.85]:
        for z in [17.54,38.54]:
            mask=(abs(uv[:,0]-a)<1.4)&(abs(uv[:,1]-z)<1.4)
            p=uv[mask]
            # Restrict to circular arcs, excluding the adjacent rectangular opening.
            mask2=abs(np.linalg.norm(p-[a,z],axis=1)-1.2)<.15
            p=p[mask2]
            fit=np.linalg.lstsq(np.column_stack((2*p,np.ones(len(p)))),(p*p).sum(1),rcond=None)[0]
            centers.append(fit[:2])
    centers=np.array(centers)
    # Center the camera on the adapter's front plate, independently of the
    # small offset of the mounting-slot pattern in the supplied STL.
    plate_center_uv=(uv.min(0)+uv.max(0))/2
    u,z=plate_center_uv
    d=(plane_vertices@n).mean()
    rear_vertices=mount.vertices[np.unique(mount.facets_boundary[1])]
    rear_d=(rear_vertices@n).mean()
    pcb_clearance_mm=5.0
    # The board mesh occupies local X=-1.6..0 mm. Its lens-facing surface
    # therefore sits exactly 5 mm behind the mount's rear face.
    pcb_stl=n*(rear_d-pcb_clearance_mm)+up*u+np.array([0,0,z])
    plate_front_stl=n*d+up*u+np.array([0,0,z])
    plate_rear_stl=n*rear_d+up*u+np.array([0,0,z])
    # Local X looks toward the fingers; local Y spans the 25 mm board dimension.
    across=(centers[1]-centers[0]+centers[3]-centers[2])/2
    across=across/np.linalg.norm(across)
    camera_left=up*across[0]+np.array([0,0,across[1]])
    camera_up=np.cross(n,camera_left)
    camera_stl_r=np.column_stack((n,camera_left,camera_up))
    camera_r=mount_r@camera_stl_r
    camera_p=(mount_r@pcb_stl+mount_t)*.001
    # Check an entire 16 x 16 mm lens-housing footprint through the actual STL
    # plate. Rays must encounter no mount triangles between its rear/front faces.
    triangles=mount.triangles
    e1=triangles[:,1]-triangles[:,0]; e2=triangles[:,2]-triangles[:,0]
    h=np.cross(n,e2); determinant=np.einsum('ij,ij->i',e1,h)
    valid=abs(determinant)>1e-9
    inv=np.zeros(len(determinant)); inv[valid]=1/determinant[valid]
    for y in np.linspace(-8,8,9):
        for z1 in np.linspace(-8,8,9):
            ray_origin=pcb_stl+camera_left*y+camera_up*z1
            s=ray_origin-triangles[:,0]
            u_hit=np.einsum('ij,ij->i',s,h)*inv
            q=np.cross(s,e1)
            v_hit=(q@n)*inv
            distance=np.einsum('ij,ij->i',e2,q)*inv
            blocked=valid&(u_hit>=0)&(v_hit>=0)&(u_hit+v_hit<=1)&(distance>=pcb_clearance_mm-1e-4)&(distance<=d-rear_d+pcb_clearance_mm+1e-4)
            if blocked.any():
                raise RuntimeError('Lens housing intersects the camera mount opening')
    v,f=plate_with_holes(25,24,1.6,[(y,z,1.1) for y in [-10.5,10.5] for z in [-6.75,6.75]])
    write_obj(OUT/"imx335_b_board.obj",np.asarray(v)*.001,np.asarray(f))
    expected_holes=np.array([camera_left*y+camera_up*z1 for y,z1 in [(-10.5,-6.75),(10.5,-6.75),(-10.5,6.75),(10.5,6.75)]])
    actual_holes=np.array([up*(p[0]-u)+np.array([0,0,p[1]-z]) for p in centers])
    data=dict(meshes=manifest,
        joints=[dict(name=name,pivot_m=(np.array(p)*.001).tolist(),axis=axis) for name,p,axis in [
            ("shoulder_pan",[-.099,-7.543,62.43],[0,-.0087265,.9999619]),
            ("shoulder_lift",[2.38,-38.72,119.758],[.9999619,.0087265,0]),
            ("elbow_flex",[4.45,73.73,148.62],[.9999238,.0087262,.0087265]),
            ("wrist_flex",[6.9,-60.577,152.59],[.9998477,0,.0174524]),
            ("wrist_roll",[7.062,-118.775,152.082],[0,-.9999619,-.0087265]),
            ("gripper",[6.07,-145.704,171.911],[.9999619,0,.0087265])]],
        wrist_camera=dict(board_center_m=camera_p.tolist(),rotation=camera_r.tolist(),
            optical_offset_m=.019,board_size_mm=[25,24],hole_pattern_mm=[21,13.5],
            placement='centered; PCB front face 5 mm behind mount rear face; lens through opening',
            pcb_front_to_mount_rear_m=.005,mount_plate_thickness_m=float((d-rear_d)*.001),
            mount_front_center_m=((mount_r@plate_front_stl+mount_t)*.001).tolist(),
            mount_rear_center_m=((mount_r@plate_rear_stl+mount_t)*.001).tolist(),
            lens_front_protrusion_m=float(.019-(d-rear_d+pcb_clearance_mm)*.001)),
        alignment=dict(gripper_bore_fit_max_error_mm=float(np.linalg.norm(source@ref_r.T+ref_t-target,axis=1).max()),
            camera_plate_center_uv_mm=plate_center_uv.tolist(),
            camera_plate_center_error_mm=float(np.linalg.norm(np.array([pcb_stl@up,pcb_stl[2]])-plate_center_uv)),
            lens_housing_clears_mount_opening=True,
            camera_board_hole_fit_max_error_mm=float(np.linalg.norm(expected_holes-actual_holes,axis=1).max()),
            dovetail_bore_fit_max_error_mm=float(np.linalg.norm(bore_errors,axis=1).max()),
            rigid_gripper_dovetail_revision_correction_mm=correction.tolist(),
            stl_to_assembly_rotation=mount_r.tolist(),stl_to_assembly_translation_mm=mount_t.tolist(),
            stl_camera_slot_centers_uv_mm=centers.tolist(),
            reference_step_camera_hole_pattern_mm=[28,28],
            note="STL camera plate supersedes the STEP reference camera plate; no adapter is used."),
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ASSETS/"cad").iterdir())})
    (ASSETS/"scene_assets.json").write_text(json.dumps(data,indent=2)+"\n")
    print("Exported",len(manifest),"arm mesh groups and wrist mount.")
    print(json.dumps(data["alignment"],indent=2))

if __name__=="__main__":
    main()
