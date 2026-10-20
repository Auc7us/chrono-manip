#!/usr/bin/env python3
"""SO-101 in a five-sided white lightbox, with two real Chrono Sensor cameras.

SI units, Z up, X across the box, Y from the open front toward the back.
The first five joints follow prescribed CAD trajectories. Automated gripping
uses detailed contacts and a torque-limited physical jaw; dynamics parameters
are not calibrated to hardware. Run ./run.sh --help for options.
"""
from pathlib import Path
import argparse
import json
import math
import time
import numpy as np
from so101_ik import START_POSES_DEG, tool_pose, forward_kinematics as cad_forward_kinematics
from scene_config import INCH,WIDTH,DEPTH,HEIGHT,SHEET_SIZE,SHEET_CENTER,WORK_SURFACE_Z
from reference_camera import DEFAULT_PROFILE,PHOTO_FIT_PROFILE,load_profile,fitted_board_pose,camera_metadata
from appearance import (COLORS, LIGHT_KELVIN, WHITE_BALANCE_KELVIN, LIGHT_BRIGHTNESS,
    AMBIENT_FILL, TUBE_CENTER, TUBE_NORMAL, TUBE_U, TUBE_V, TUBE_FRAME, EMITTER_CENTER,
    EMITTER_SIZE, quadrature, node_strength, floor_bounce_nodes, ceiling_bounce_nodes, appearance_metadata)

ROOT = Path(__file__).resolve().parent
REFERENCE_LAYOUT=json.loads((ROOT/'assets/reference_layout.json').read_text())
DEFAULT_JOINTS = REFERENCE_LAYOUT['inspection_joint_deg']
CUBE_EDGE, CUBE_MASS = .030, .030  # 3 cm PLA cube, 30 grams
TRAY_SIZE, TRAY_RIM_HEIGHT = .050, .010  # 5 cm square, 1 cm above its inside floor
TRAY_BASE, TRAY_WALL = .002, .002
TRAY_CENTER = tuple(REFERENCE_LAYOUT['tray_center_xy_m'])

def unit(v):
    v=np.asarray(v,dtype=float)
    return v/np.linalg.norm(v)

def axis_rotation(axis, angle):
    a=unit(axis)
    x,y,z=a
    k=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
    return np.eye(3)+math.sin(angle)*k+(1-math.cos(angle))*(k@k)

def transform(r=None,p=None):
    t=np.eye(4)
    if r is not None: t[:3,:3]=r
    if p is not None: t[:3,3]=p
    return t

def around(pivot, axis, degrees):
    r=axis_rotation(axis,math.radians(degrees))
    p=np.asarray(pivot)
    return transform(r,p-r@p)

def forward_kinematics(assets, angles):
    # The base footprint's front edge meets Y=0; its center stays on X=0.
    # The input assembly reaches toward -Y, hence a 180-degree mounting rotation.
    return cad_forward_kinematics(assets,angles)

def point(t,p):
    return t[:3,:3]@np.asarray(p)+t[:3,3]

def view_rotation(direction, up=(0,0,1)):
    forward=unit(direction)
    left=unit(np.cross(up,forward))
    return np.column_stack((forward,left,np.cross(forward,left)))

def global_camera_mount(config,profile=None,optical_offset=.019):
    """Return the fixed camera name and PCB pose in the lightbox frame."""
    if config=='top-down':
        return 'top_down',transform(view_rotation([0,0,-1],[0,1,0]),[0,DEPTH/2,HEIGHT-.007])
    if config=='right-45':
        return 'right_45',fitted_board_pose(profile,optical_offset)
    raise ValueError(f'Unknown global camera configuration: {config}')

def blackbody_rgb(kelvin):
    # Approximate blackbody chromaticity; turn sRGB into linear RGB for rendering.
    t=kelvin/100
    r=255 if t<=66 else 329.698727446*(t-60)**-.1332047592
    g=99.4708025861*math.log(t)-161.1195681661 if t<=66 else 288.1221695283*(t-60)**-.0755148492
    b=255 if t>=66 else (0 if t<=19 else 138.5177312231*math.log(t-10)-305.0447927307)
    rgb=np.clip(np.array([r,g,b])/255,0,1)
    return np.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055)**2.4)

def white_balanced_light(args):
    rgb=blackbody_rgb(args.light_kelvin)/blackbody_rgb(args.white_balance_kelvin)
    return rgb/np.dot(rgb,[.2126,.7152,.0722])

def parse_args():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--motion',action='store_true',help='slow, small joint sweep around the initial pose')
    p.add_argument('--joint-deg',type=float,nargs=6,default=DEFAULT_JOINTS,metavar=('PAN','LIFT','ELBOW','WRIST','ROLL','JAW'))
    p.add_argument('--assembly-pose',action='store_true',help='inspect the original STEP pose (overrides --joint-deg)')
    p.add_argument('--headless',action='store_true',help='render sensors without opening windows')
    p.add_argument('--no-sensors',action='store_true',help='third-person CAD inspection only; no sensor manager/GPU')
    p.add_argument('--duration',type=float,default=None,help='stop after this many simulation seconds')
    p.add_argument('--width',type=int,default=640,help='sensor image width (4:3 aspect ratio)')
    p.add_argument('--rate',type=float,default=15,help='both cameras update at this rate in Hz')
    p.add_argument('--global-camera',choices=['top-down','right-45'],default='right-45',
                   help='fixed camera: nominal centered right-wall 45-degree view (default), or optional virtual top-down')
    p.add_argument('--camera-profile',type=Path,default=DEFAULT_PROFILE,help='JSON with fixed optical pose and fitted lens parameters')
    p.add_argument('--lens-model',choices=['reference','pinhole'],default='reference',help='fitted wide-angle projection (default), or rectified pinhole')
    p.add_argument('--raw-response',action='store_true',help='skip the fixed photographic exposure/contrast response for fitting comparisons')
    p.add_argument('--samples',type=int,default=2,help='spatial samples per axis (2 means a fixed 2 x 2 subpixel grid)')
    p.add_argument('--denoiser',choices=['on','off'],default='off',help='OptiX denoiser for optional stochastic modes only')
    p.add_argument('--record-dir',type=Path,help='record consecutive camera PNG frames and poses (requires PIL)')
    p.add_argument('--hfov-deg',type=float,default=110,help='rectified horizontal FOV; not the 175-degree diagonal fisheye rating')
    p.add_argument('--exposure',type=float,default=1.,help='radiometric display exposure multiplier')
    p.add_argument('--light-kelvin',type=float,default=LIGHT_KELVIN,help='GC21B color temperature; photographic estimate: 4600 K')
    p.add_argument('--white-balance-kelvin',type=float,default=WHITE_BALANCE_KELVIN,help='approximate fixed camera white balance: 4800 K')
    p.add_argument('--light-brightness',type=float,default=LIGHT_BRIGHTNESS,help='GC21B dimmer fraction, 0 to 1; default 0.5')
    rendering=p.add_mutually_exclusive_group()
    rendering.add_argument('--fast',action='store_true',help='tube-center point-light approximation with harder shadows; no denoising')
    rendering.add_argument('--path-traced',action='store_true',help='optional stochastic diffuse bounce; use more samples to reduce grain')
    rendering.add_argument('--stochastic-area',action='store_true',help='optional random area-light sampling for renderer comparisons')
    p.add_argument('--save-dir',type=Path,help='save the final wrist/global frames and camera poses here')
    p.add_argument('--validate',action='store_true',help='check assets/geometry without importing PyChrono')
    p.add_argument('--auto',action='store_true',help='IK pick-and-place episodes with physical jaw contacts and torque-limited gripping')
    p.add_argument('--start-pose',choices=list(START_POSES_DEG),default='folded',
                   help='automated start/return pose: folded resting position (default), or fully extended to the left side')
    p.add_argument('--episodes',type=int,default=1,help='number of automated episodes')
    p.add_argument('--seed',type=int,default=42,help='reproducible cube/tray randomization seed')
    p.add_argument('--tray-every',type=int,default=5,help='move the red tray every N episodes (0 keeps its first sampled pose)')
    p.add_argument('--fixed-cube',action='store_true',help='use one fixed cube spawn rather than randomizing each episode')
    p.add_argument('--replay-episode',type=Path,help='reuse the cube/tray layout from an episode.json with the current planner and physics')
    p.add_argument('--collect-dir',type=Path,help='new directory for synchronized episode images, videos, and trajectories')
    p.add_argument('--no-realtime',action='store_true',help='run as fast as physics/rendering allows')
    a=p.parse_args()
    if a.width<64 or a.rate<=0 or a.samples<1 or not 10<a.hfov_deg<170 or a.exposure<=0:
        p.error('width >= 64, rate > 0, samples >= 1, 10 < hfov-deg < 170, exposure > 0 required')
    if not 2700<=a.light_kelvin<=6500 or not 2000<=a.white_balance_kelvin<=10000 or not 0<=a.light_brightness<=1:
        p.error('light-kelvin must be 2700–6500, white-balance-kelvin 2000–10000, light-brightness 0–1')
    if a.duration is not None and a.duration<=0: p.error('--duration must be positive')
    if a.episodes<1 or a.tray_every<0:p.error('episodes must be positive and tray-every nonnegative')
    if a.auto and (a.motion or a.assembly_pose):p.error('--auto cannot be combined with --motion or --assembly-pose')
    if a.replay_episode and not a.auto:p.error('--replay-episode requires --auto')
    if a.collect_dir and (not a.auto or a.no_sensors):p.error('--collect-dir requires --auto and both sensors')
    if a.collect_dir and abs(a.rate-round(a.rate))>1e-9:p.error('collection rate must be an integer FPS')
    if a.headless and a.duration is None and not a.auto: a.duration=3.
    return a

def validate(assets):
    for part in assets['meshes']+[{'file':name} for name in ['wrist_camera_mount.obj','imx335_b_board.obj',
            'floor_sheet_detail.obj','gc21b_diffuser.obj','gc21b_black.obj','adapter_screw_head.obj',
            'panel_floor.obj','panel_ceiling.obj','panel_left.obj','panel_right.obj','panel_rear.obj']]:
        path=ROOT/'assets/meshes'/part['file']
        if not path.is_file(): raise RuntimeError(f'Missing asset: {path}; see README.md')
    jaw_geometry=json.loads((ROOT/'assets/jaw_collision.json').read_text())
    for part in jaw_geometry['parts'].values():
        if not (ROOT/'assets/meshes'/part['file']).is_file():raise RuntimeError(f'Missing detailed jaw collision mesh: {part["file"]}')
    assert len(assets['joints'])==6
    assert assets['alignment']['gripper_bore_fit_max_error_mm']<.01
    assert assets['alignment']['camera_plate_center_error_mm']<1e-5
    assert assets['alignment']['dovetail_bore_fit_max_error_mm']<.01
    cam=assets['wrist_camera']
    forward=np.array(cam['rotation'])[:,0]
    clearance=np.dot(np.array(cam['mount_rear_center_m'])-cam['board_center_m'],forward)
    protrusion=np.dot(np.array(cam['board_center_m'])+forward*cam['optical_offset_m']-cam['mount_front_center_m'],forward)
    assert abs(clearance-.005)<1e-8
    assert protrusion>0 and abs(protrusion-cam['lens_front_protrusion_m'])<1e-8
    assert assets['alignment']['lens_housing_clears_mount_opening']
    for frame in forward_kinematics(assets,DEFAULT_JOINTS):
        assert np.allclose(frame[:3,:3].T@frame[:3,:3],np.eye(3),atol=1e-7)
    for name in ['white','floor_sheet']:
        if not (ROOT/'assets/textures'/f'{name}.png').is_file():
            raise RuntimeError('Missing surface textures; run python3 tools/prepare_appearance.py')
    print(f'Interior: {WIDTH:.4f} W × {DEPTH:.4f} D × {HEIGHT:.4f} H m (35 × 24 × 20 inches)')
    print('Five opaque walls, open front at Y=0; base clamped at front center.')
    print('CAD bore fit: %.6f mm; camera plate center offset: %.6f mm' % (
        assets['alignment']['gripper_bore_fit_max_error_mm'],assets['alignment']['camera_plate_center_error_mm']))
    print(f'Nominal CAD PCB clearance: {clearance*1000:.3f} mm; lens protrusion: {protrusion*1000:.3f} mm (before photo-fit correction)')

class Scene:
    def __init__(self,chrono,assets,args,simulate_jaw=True):
        self.c=chrono; self.assets=assets; self.args=args
        self.jaw_physics=None;self._last_arm_time=None
        self.system=chrono.ChSystemNSC()
        self.system.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)
        self.system.GetSolver().AsIterative().SetMaxIterations(150)
        self.system.SetGravitationalAcceleration(chrono.ChVector3d(0,0,-9.81))
        chrono.ChCollisionModel.SetDefaultSuggestedEnvelope(.00005)
        chrono.ChCollisionModel.SetDefaultSuggestedMargin(.00005)
        self.materials={name:self.material(rgb) for name,rgb in COLORS.items()}
        # Lightbox panels and printed PLA are matte; localized glossy responses
        # should not create a bright patch on the white working surface.
        for name in ['white','cream','pla','red','diffuser','floor_sheet','black']:
            mat=self.materials[name]
            mat.SetSpecularColor(chrono.ChColor(0,0,0))
            mat.SetRoughness(.92 if name in ['floor_sheet','black'] else .86)
            mat.SetUseSpecularWorkflow(True)
        self.materials['floor_sheet'].SetSpecularColor(chrono.ChColor(.018,.018,.018))
        self.materials['black'].SetSpecularColor(chrono.ChColor(.025,.025,.025))
        for name in ['white','floor_sheet']:
            self.materials[name].SetKdTexture(str(ROOT/'assets/textures'/f'{name}.png'))
        rgb=white_balanced_light(args)
        self.materials['diffuser'].SetEmissiveColor(chrono.ChColor(*map(float,rgb/rgb.max())))
        self.materials['diffuser'].SetEmissivePower(float(4*args.light_brightness*args.exposure*rgb.max()))
        self.materials['metal'].SetMetallic(.96)
        self.materials['metal'].SetSpecularColor(chrono.ChColor(.72,.74,.77))
        self.materials['metal'].SetRoughness(.14)
        self.materials['rack_metal'].SetSpecularColor(chrono.ChColor(.14,.14,.14))
        self.materials['rack_metal'].SetRoughness(.22)
        self.materials['polished_metal'].SetSpecularColor(chrono.ChColor(.90,.92,.95))
        self.materials['polished_metal'].SetRoughness(.095)
        self.materials['polished_metal'].SetMetallic(.98)
        self.materials['polished_metal'].SetUseSpecularWorkflow(False)
        self.links=[self.body(f'SO101 link {i}',aux_ref=i in [5,6]) for i in range(7)]
        jaw_geometry=json.loads((ROOT/'assets/jaw_collision.json').read_text())['parts']
        for name in ['fixed_jaw','moving_jaw']:
            part=jaw_geometry[name]
            self.links[part['link']].SetFrameCOMToRef(self.frame(transform(p=part['center_of_mass_ref_m'])))
        self.world=self.body('Lightbox and fixed global camera')
        for part in assets['meshes']:
            self.mesh(self.links[part['link']],part['file'],part['material'])
        self.mesh(self.links[5],'wrist_camera_mount.obj','black')
        # Longer fasteners sit on the adapter face. Original STEP heads are
        # below this STL plate and would otherwise appear as dark empty holes.
        for y,z in [(-.14548569,.14720039),(-.14555698,.15529884)]:
            self.cylinder(self.links[5],.0028,.00035,[.03535,y,z],'metal',(1,0,0))
            self.mesh(self.links[5],'adapter_screw_head.obj','polished_metal',transform(p=[.03545,y,z]))
            self.box(self.links[5],(.00008,.0032,.00065),(.03555,y,z),'screw_recess')
            self.box(self.links[5],(.00008,.00065,.0032),(.03555,y,z),'screw_recess')
        self.make_box()
        self.make_clamps()
        cam=assets['wrist_camera']
        self.wrist_board=transform(np.array(cam['rotation']),cam['board_center_m'])
        lens=transform(p=[cam['optical_offset_m'],0,0])
        image_frame=lens@transform(np.diag([1.,-1.,-1.]))
        self.wrist_fit=json.loads((ROOT/'assets/reference_wrist.json').read_text())
        self.wrist_optical=self.wrist_board@image_frame
        if args.lens_model=='reference':
            self.wrist_optical=self.wrist_optical@np.array(self.wrist_fit['optical_from_corrected_optical'])
            # Place the modeled PCB/lens at the same fitted frame as the sensor,
            # keeping rays clear of its own approximated lens barrel.
            self.wrist_board=self.wrist_optical@np.linalg.inv(image_frame)
        self.camera_model(self.links[5],self.wrist_board,cam)
        self.camera_profile=load_profile(args.camera_profile)
        self.global_camera_name,self.global_board=global_camera_mount(args.global_camera,self.camera_profile,cam['optical_offset_m'])
        self.camera_names=['wrist',self.global_camera_name]
        self.camera_model(self.world,self.global_board)
        if args.global_camera=='right-45':
            self.box(self.world,(.012,.034,.028),(WIDTH/2-.006,self.global_board[1,3],HEIGHT-.004),'black')
        # The CAD mounting frame puts the imager upside down. Roll its view
        # 180 degrees about the optical axis, preserving its position and aim.
        self.global_optical=self.global_board@lens
        self.make_object_and_tray()
        self.tcp_local=np.linalg.inv(forward_kinematics(assets,[0]*6)[5])@tool_pose(assets,np.zeros(6))
        self.cables=[]
        self.update_arm(0)
        self.make_cables()
        from jaw_physics import JawPhysics
        self.jaw_physics=JawPhysics(self,dynamic=simulate_jaw and args.auto)

    def vec(self,p): return self.c.ChVector3d(*map(float,p))

    def quaternion(self,r):
        # Construct a unit quaternion from a rotation matrix, including 180° cases.
        tr=np.trace(r)
        if tr>0:
            s=math.sqrt(tr+1)*2
            q=[s/4,(r[2,1]-r[1,2])/s,(r[0,2]-r[2,0])/s,(r[1,0]-r[0,1])/s]
        else:
            i=int(np.argmax(np.diag(r))); j=(i+1)%3; k=(i+2)%3
            s=math.sqrt(1+r[i,i]-r[j,j]-r[k,k])*2
            xyz=[0.,0.,0.]; xyz[i]=s/4; xyz[j]=(r[j,i]+r[i,j])/s; xyz[k]=(r[k,i]+r[i,k])/s
            q=[(r[k,j]-r[j,k])/s,*xyz]
        return self.c.ChQuaterniond(*q)

    def frame(self,t): return self.c.ChFramed(self.vec(t[:3,3]),self.quaternion(t[:3,:3]))

    def material(self,rgb):
        m=self.c.ChVisualMaterial()
        m.SetDiffuseColor(self.c.ChColor(*map(float,rgb)))
        m.SetAmbientColor(self.c.ChColor(*map(float,np.array(rgb)*.3)))
        m.SetSpecularColor(self.c.ChColor(.12,.12,.12))
        m.SetRoughness(.72); m.SetMetallic(0.)
        return m

    def body(self,name,aux_ref=False):
        b=self.c.ChBodyAuxRef() if aux_ref else self.c.ChBody()
        b.SetName(name); b.SetFixed(True); self.system.Add(b)
        return b

    def shape(self,body,shape,mat,t=None):
        shape.AddMaterial(self.materials[mat])
        body.AddVisualShape(shape,self.frame(np.eye(4) if t is None else t))
        return shape

    def box(self,body,size,pos,mat,r=None):
        return self.shape(body,self.c.ChVisualShapeBox(*map(float,size)),mat,transform(r,pos))

    def cylinder(self,body,radius,length,pos,mat,axis=(0,0,1)):
        z=unit(axis); guide=[0,1,0] if abs(z[1])<.9 else [1,0,0]
        x=unit(np.cross(guide,z)); y=np.cross(z,x)
        return self.shape(body,self.c.ChVisualShapeCylinder(radius,length),mat,transform(np.column_stack((x,y,z)),pos))

    def mesh(self,body,name,mat,t=None):
        mesh=self.c.ChTriangleMeshConnected.CreateFromWavefrontFile(str(ROOT/'assets/meshes'/name),True,True)
        shape=self.c.ChVisualShapeTriangleMesh(); shape.SetMesh(mesh); shape.SetBackfaceCull(False)
        return self.shape(body,shape,mat,t)

    def make_box(self):
        w,d,h,t=WIDTH,DEPTH,HEIGHT,.006
        for name in ['floor','ceiling','left','right','rear']:
            self.mesh(self.world,f'panel_{name}.obj','white')
        self.mesh(self.world,'floor_sheet_detail.obj','floor_sheet')
        self.diffuser_center=tuple(EMITTER_CENTER)
        self.diffuser_size=EMITTER_SIZE
        for material in ['black','diffuser']:
            self.mesh(self.world,f'gc21b_{material}.obj',material,transform(TUBE_FRAME,TUBE_CENTER))
        if self.args.global_camera=='top-down':
            self.box(self.world,(.034,.033,.004),(0,d/2,h-.002),'black')
        # Visible board joins and exposed thin front edge. These are visual
        # estimates; the white panels retain their original collision plane.
        # Exposed board cores sit below the white face; uneven wall gaps remain
        # open instead of being covered by a full opaque backing box.
        for x in [-w/2,w/2]:
            self.box(self.world,(.0016,d,.004),(x,d/2,-.0004),'panel_edge')
        self.box(self.world,(w,.0016,.004),(0,d,-.0004),'panel_edge')
        self.box(self.world,(w,.0016,.004),(0,-.001,-.0004),'panel_edge')
        # Small overlapping seam tabs visible near the front two corners.
        for x in [-w/2+.006,w/2-.006]:
            self.box(self.world,(.011,.034,.00022),(x,.020,.0019),'tape')
        self.make_room_background()
        # A small exterior base below the box makes the clamp's underside visible.
        self.box(self.world,(w+.32,d+.13,.018),(.035,d/2-.012,-.014),'white')

    def make_room_background(self):
        # Only the open front sees this dim room. Approximate large structures
        # from the reference, without presenting their dimensions as measured.
        self.box(self.world,(3.2,.04,2.1),(0,-1.4,.58),'room_wall')
        self.box(self.world,(3.2,2.2,.03),(0,-.65,-.20),'room_floor')
        self.box(self.world,(.025,1.65,1.45),(-.94,-.84,.30),'room_wall')
        for x in [-1.6,1.6]:
            self.box(self.world,(.04,3.2,2.1),(x,-.4,.58),'room_wall')
        self.box(self.world,(3.2,3.2,.04),(0,-.4,1.65),'room_wall')
        for x in [-.68,-.28]:
            self.box(self.world,(.025,.26,1.15),(x,-.95,.37),'room_dark')
        for z in [-.14,.15,.44,.73,.93]:
            self.box(self.world,(.43,.26,.025),(-.48,-.95,z),'room_dark')
        for row,z in enumerate([.01,.30,.59,.83]):
            for i in range(8):
                x=-.65+i*.047
                material=['room_blue','room_green','room_book','room_paper','room_dark'][(i+3*row)%5]
                self.box(self.world,(.025+.004*(i%3),.16,.17+.015*((i+row)%4)),
                         (x,-.94,z),material,axis_rotation([0,1,0],.025*((i+row)%3-1)))
                self.box(self.world,(.014,.0007,.004),(x,-.859,z+.035),'room_paper')
        self.make_reference_room_details()

    def make_reference_room_details(self):
        # Depth estimates for image-observed features. These remain 3D geometry
        # when viewed from the wrist or optional virtual overhead camera.
        from reference_camera import pixel_rays
        profile=load_profile(PHOTO_FIT_PROFILE);origin=np.array(profile['optical_position_m'])
        def room_points(pixels,axis,value):
            rays=pixel_rays(pixels,profile)
            return origin+rays*((value-origin[axis])/rays[:,axis])[:,None]
        def wire(pixels,axis,value,radius=.002):
            points=room_points(pixels,axis,value)
            for a,b in zip(points,points[1:]):
                self.cylinder(self.world,radius,np.linalg.norm(b-a),(a+b)/2,'rack_metal',b-a)
        # Two tubular handles, basket rim and fine welded wire rails.
        wire([(83,114),(82,106),(86,96),(99,90),(106,90),(110,96),(112,118)],0,-.32)
        wire([(150,113),(150,106),(157,99),(168,97),(174,100),(174,119)],0,-.62)
        for y in [121,128,136,145,155,165]:
            wire([(55,y+5),(93,y),(133,y-1),(174,y-2)],1,-.20,.0012)
        for x in range(58,175,10):wire([(x,121),(x-1,164),(x-4,190)],1,-.20,.0010)
        # Faint wall outlet next to the bookshelf.
        center=room_points([[96,41]],0,-.922)[0]
        self.box(self.world,(.003,.037,.055),center,'room_paper')
        for dy in [-.008,.008]:
            for dz in [-.010,.010]:
                self.box(self.world,(.0035,.002,.005),center+[.001,dy,dz],'room_dark')
        # Rust-colored chair seat glimpsed outside the front left of the image.
        points=room_points([[0,344],[38,366],[20,411],[0,402]],2,-.16)
        center=points.mean(0);side_a=points[1]-points[0];side_b=points[3]-points[0]
        u=unit(side_a);v=unit(np.cross([0,0,1],u))
        self.box(self.world,(np.linalg.norm(side_a),np.linalg.norm(side_b),.025),center,'room_rust',
                 np.column_stack([u,v,[0,0,1.]]))

    def make_clamps(self):
        # Two front-edge screw clamps engage the actual base feet and box support.
        for x in [-.045,.045]:
            self.box(self.world,(.017,.064,.009),(x,.016,.010),'black')
            self.box(self.world,(.017,.012,.068),(x,-.014,-.017),'black')
            self.box(self.world,(.017,.045,.009),(x,.007,-.049),'black')
            self.cylinder(self.world,.0025,.043,(x,.026,-.030),'metal')
            self.cylinder(self.world,.007,.003,(x,.026,-.010),'black')
            self.cylinder(self.world,.002,.026,(x,.026,-.054),'metal',(1,0,0))

    def camera_model(self,body,pose,mount=None):
        self.mesh(body,'imx335_b_board.obj','pcb',pose)
        # M12 lens housing, barrel, retaining ring, lens glass, USB connector.
        self.box(body,(.006,.016,.016),point(pose,[.003,0,0]),'black',pose[:3,:3])
        for x,r,length,mat in [(.010,.006,.009,'black'),(.015,.0072,.003,'black'),(.018,.0057,.002,'glass')]:
            self.cylinder(body,r,length,point(pose,[x,0,0]),mat,pose[:3,0])
        self.box(body,(.005,.008,.003),point(pose,[-.004,0,-.009]),'cable_white',pose[:3,:3])
        for y in [-.0105,.0105]:
            for z in [-.00675,.00675]:
                self.cylinder(body,.0018,.0006,point(pose,[.0005,y,z]),'gold',pose[:3,0])
                if mount:
                    gap=mount['pcb_front_to_mount_rear_m']
                    front=gap+mount['mount_plate_thickness_m']
                    self.cylinder(body,.0015,gap,point(pose,[gap/2,y,z]),'metal',pose[:3,0])
                    self.cylinder(body,.0009,front+.002,point(pose,[(front-.002)/2,y,z]),'metal',pose[:3,0])
                    self.cylinder(body,.0018,.001,point(pose,[front+.0005,y,z]),'metal',pose[:3,0])
                else:
                    self.cylinder(body,.0009,.004,point(pose,[-.0005,y,z]),'metal',pose[:3,0])
        # Simple USB strain relief leaving the connector along the camera board.
        self.box(body,(.004,.006,.006),point(pose,[-.004,0,-.014]),'black',pose[:3,:3])

    def make_object_and_tray(self):
        # The 5 x 5 cm measurement is the tray's outside footprint. Its raised
        # edge is 1 cm above the inside floor; base and wall thickness are 2 mm.
        c=self.c
        contact=c.ChContactMaterialNSC()
        contact.SetFriction(.4); contact.SetRestitution(0.)
        self.world.EnableCollision(True)
        self.world.AddCollisionShape(c.ChCollisionShapeBox(contact,WIDTH,DEPTH,.006),
            self.frame(transform(p=[0,DEPTH/2,-.003])))
        self.world.AddCollisionShape(c.ChCollisionShapeBox(contact,*SHEET_SIZE),
            self.frame(transform(p=SHEET_CENTER)))
        self.tray=self.body('Red placement tray: 5 x 5 cm, 1 cm rim')
        self.tray.SetPos(self.vec([*TRAY_CENTER,WORK_SURFACE_Z]))
        self.tray.SetRot(self.quaternion(axis_rotation([0,0,1],REFERENCE_LAYOUT['tray_yaw_rad'])))
        self.tray.EnableCollision(True)
        def tray_box(size,pos):
            self.box(self.tray,size,pos,'red')
            self.tray.AddCollisionShape(c.ChCollisionShapeBox(contact,*size),self.frame(transform(p=pos)))
        tray_box((TRAY_SIZE,TRAY_SIZE,TRAY_BASE),(0,0,TRAY_BASE/2))
        edge=(TRAY_SIZE-TRAY_WALL)/2
        wall_z=TRAY_BASE+TRAY_RIM_HEIGHT/2
        for side in [-1,1]:
            tray_box((TRAY_WALL,TRAY_SIZE,TRAY_RIM_HEIGHT),(side*edge,0,wall_z))
            tray_box((TRAY_SIZE-2*TRAY_WALL,TRAY_WALL,TRAY_RIM_HEIGHT),(0,side*edge,wall_z))
        # Set the requested printed-part mass, rather than assuming solid PLA
        # density. A uniform cube inertia is the approximation used here.
        self.cube=c.ChBodyEasyBox(CUBE_EDGE,CUBE_EDGE,CUBE_EDGE,CUBE_MASS/CUBE_EDGE**3,True,True,contact)
        self.cube.SetName('PLA cube: 3 x 3 x 3 cm, 30 g')
        self.cube.SetMass(CUBE_MASS)
        self.cube.SetInertiaXX(self.vec([CUBE_MASS*CUBE_EDGE**2/6]*3))
        self.cube.SetPos(self.vec([*TRAY_CENTER,WORK_SURFACE_Z+TRAY_BASE+CUBE_EDGE/2+.0002]))
        self.cube.SetRot(self.quaternion(axis_rotation([0,0,1],REFERENCE_LAYOUT['cube_yaw_rad'])))
        self.cube.GetVisualShape(0).SetMaterial(0,self.materials['pla'])
        self.system.Add(self.cube)
        assert abs(self.cube.GetMass()-.030)<1e-12
        assert CUBE_EDGE < TRAY_SIZE-2*TRAY_WALL

    def cube_pose(self):
        p=self.cube.GetPos(); q=self.cube.GetRot()
        return [p.x,p.y,p.z,q.e0,q.e1,q.e2,q.e3]

    def camera_world_pose(self,name):
        return self.body_transform(self.links[5])@self.wrist_optical if name=='wrist' else self.global_optical

    def tool_world_pose(self):
        return self.body_transform(self.links[5])@self.tcp_local

    def tray_pose(self):
        p=self.tray.GetPos();q=self.tray.GetRot()
        return [p.x,p.y,p.z,q.e0,q.e1,q.e2,q.e3]

    def body_transform(self,body):
        ref=body.GetFrameRefToAbs()
        q=ref.GetRot();w,x,y,z=q.e0,q.e1,q.e2,q.e3
        r=np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                    [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                    [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
        p=ref.GetPos()
        return transform(r,[p.x,p.y,p.z])

    def shared_state(self):
        return list(self.joint_deg)+self.cube_pose()+self.tray_pose()

    def apply_shared_state(self,t,state):
        self.update_arm(t,state[:6]);self.set_cube_pose(state[6:13])
        self.tray.SetPos(self.vec(state[13:16]));self.tray.SetRot(self.c.ChQuaterniond(*state[16:20]))

    def set_cube_pose(self,pose):
        self.cube.SetPos(self.vec(pose[:3]))
        self.cube.SetRot(self.c.ChQuaterniond(*pose[3:]))

    def save_object_state(self,output):
        output.mkdir(parents=True,exist_ok=True)
        data=dict(appearance=appearance_metadata(self.args),inspection_joint_deg=list(self.joint_deg),
                  work_surface=dict(color='black',dimensions_m=list(SHEET_SIZE),center_m=list(SHEET_CENTER),top_z_m=WORK_SURFACE_Z),
                  cube=dict(material='PLA',color='green',dimensions_m=[CUBE_EDGE]*3,mass_kg=self.cube.GetMass(),
                           world_pose_xyz_wxyz=self.cube_pose(),inertia_kg_m2=[CUBE_MASS*CUBE_EDGE**2/6]*3),
                  tray=dict(color='red',outer_dimensions_m=[TRAY_SIZE]*2,base_thickness_m=TRAY_BASE,
                            wall_thickness_m=TRAY_WALL,rim_height_above_floor_m=TRAY_RIM_HEIGHT,
                            inside_dimensions_m=[TRAY_SIZE-2*TRAY_WALL]*2,world_pose_xyz_wxyz=self.tray_pose()))
        (output/'object_and_tray.json').write_text(json.dumps(data,indent=2)+'\n')

    def update_arm(self,t,angles=None,reset=False):
        override=angles is not None
        angles=np.array(angles if override else ([0]*6 if self.args.assembly_pose else self.args.joint_deg),dtype=float)
        if self.args.motion and not override:
            angles+=np.array([8,4,5,4,8,8])*math.sin(2*math.pi*t/12)
        self.commanded_joint_deg=angles.copy()
        dynamic=self.jaw_physics is not None and self.jaw_physics.dynamic
        if dynamic and not reset:angles[5]=self.jaw_physics.angle_deg()
        self.joint_deg=angles.copy()
        previous=getattr(self,'frames',None)
        elapsed=t-self._last_arm_time if self._last_arm_time is not None else 0.
        self.frames=forward_kinematics(self.assets,angles)
        for i,(b,f) in enumerate(zip(self.links,self.frames)):
            if i in [5,6]:
                if dynamic and not reset:continue
                b.SetFrameRefToAbs(self.frame(f))
                b.SetPosDt(self.vec([0,0,0]));b.SetAngVelParent(self.vec([0,0,0]))
            else:
                b.SetPos(self.vec(f[:3,3])); b.SetRot(self.quaternion(f[:3,:3]))
                velocity=np.zeros(3);omega=np.zeros(3)
                if previous is not None and elapsed>1e-9 and not reset:
                    velocity=(f[:3,3]-previous[i][:3,3])/elapsed
                    r=f[:3,:3]@previous[i][:3,:3].T
                    omega=np.array([r[2,1]-r[1,2],r[0,2]-r[2,0],r[1,0]-r[0,1]])/(2*elapsed)
                b.SetPosDt(self.vec(velocity));b.SetAngVelParent(self.vec(omega))
        self._last_arm_time=t
        if dynamic:
            self.jaw_physics.set_parent_pose(self.frames[5],t,reset=reset)
            self.jaw_physics.drive(self.commanded_joint_deg[5],angles[5])
        if (hasattr(self,'cables') and self.cables and not (self.args.headless and self.args.no_sensors)
                and t>=getattr(self,'_next_cable_update',0.)-1e-8):
            self.update_cables()
            self._next_cable_update=(math.floor(t*self.args.rate+1e-6)+1)/self.args.rate

    def cable_points(self,i,strand):
        # Rest-frame connectors are on the exposed servo side. Each span joins
        # neighboring motor housings and follows them under joint motion.
        ports=np.array([[.014,.014,.044],[.022,-.038,.105],[.024,.075,.133],
                        [.025,-.045,.151],[.035,-.105,.151],[.026,-.145,.155]])
        a=point(self.frames[i],ports[i]); b=point(self.frames[i+1],ports[i+1])
        side=unit(self.frames[i][:3,:3]@np.array([1.,0,0]))
        # Add slack perpendicular to the direct span, outside the printed links.
        u=np.linspace(0,1,13)
        return (1-u[:,None])*a+u[:,None]*b+np.sin(np.pi*u)[:,None]*side*(.015+.003*i)+side*(strand-1)*.002

    def make_cables(self):
        for i in range(5):
            for strand,mat in enumerate(['red','black','cable_white']):
                points=self.cable_points(i,strand)
                for j,(a,b) in enumerate(zip(points[:-1],points[1:])):
                    body=self.body(f'Daisy chain {i+1}, {mat}, {j}')
                    length=float(np.linalg.norm(b-a))*1.08
                    self.cylinder(body,.00085,length,(0,0,0),mat)
                    self.cables.append((i,strand,j,body))
        self.update_cables()

    def update_cables(self):
        cache={}
        for i,strand,j,body in self.cables:
            key=(i,strand)
            if key not in cache: cache[key]=self.cable_points(i,strand)
            a,b=cache[key][j:j+2]; z=unit(b-a)
            guide=[0,1,0] if abs(z[1])<.9 else [1,0,0]
            x=unit(np.cross(guide,z)); y=np.cross(z,x)
            body.SetPos(self.vec((a+b)/2)); body.SetRot(self.quaternion(np.column_stack((x,y,z))))

def stochastic_render(args):
    return args.path_traced or args.stochastic_area

def use_denoiser(args):
    return stochastic_render(args) and args.denoiser=='on'

def render_mode(args):
    if args.fast: return 'fixed_tube_point'
    if args.path_traced: return 'stochastic_diffuse_bounce'
    if args.stochastic_area: return 'stochastic_area'
    return 'fixed_gc21b_tube_quadrature'

def configure_sensors(scene,sens):
    c=scene.c; a=scene.args
    from reference_camera import configure_projection
    configure_projection(sens,scene.camera_profile)
    manager=sens.ChSensorManager(scene.system)
    manager.SetRayRecursions(5)
    # Neutral white-panel bounce, scaled with the dimmer and display exposure.
    # Optional path tracing computes bounce instead of using this full fill.
    ambient=(.02 if a.path_traced else AMBIENT_FILL)*a.exposure*a.light_brightness/.5
    rgb=white_balanced_light(a)
    manager.scene.SetAmbientLight(c.ChVector3f(*map(float,rgb*ambient)))
    # Some installed SWIG builds omit the Background type; leave its default in
    # that build, because GetBackground returns an opaque, leaking SWIG pointer.
    if hasattr(sens,'Background'):
        background=sens.Background()
        background.mode=sens.BackgroundMode_SOLID_COLOR
        background.color_zenith=c.ChVector3f(.008,.010,.007)
        manager.scene.SetBackground(background)
    strength=node_strength(a.light_brightness)*a.exposure
    nodes=quadrature()
    total_strength=strength*len(nodes)
    source=c.ChVector3f(*map(float,EMITTER_CENTER))
    if a.fast:
        manager.scene.AddPointLight(source,c.ChColor(*map(float,rgb*total_strength)),-1.,False)
    elif stochastic_render(a):
        # Optional comparison mode: one area emitter avoids the installed PATH
        # shader's last-light-only accumulation issue with multiple emitters.
        # The installed rectangle sampler already includes a solid-angle PDF;
        # its radiometric normalization differs from LEGACY delta-light nodes.
        radiance=total_strength*.10
        manager.scene.AddRectangleLight(source,c.ChColor(*map(float,rgb*radiance)),-1.,
            c.ChVector3f(*map(float,TUBE_U*EMITTER_SIZE[0])),
            c.ChVector3f(*map(float,TUBE_V*EMITTER_SIZE[1])),False)
    else:
        # Fixed spatial quadrature of ONE narrow inflatable tube. LEGACY sums
        # all nodes correctly, unlike the installed PATH direct-light loop.
        # No random light samples or learned denoiser reshape the live image.
        color=c.ChColor(*map(float,rgb*strength))
        for p in nodes:
            # Broad hemispherical emission approximates the tube's diffuser;
            # its opaque backing suppresses grazing/upward spill and wall hotspots.
            manager.scene.AddSpotLight(c.ChVector3f(*map(float,p)),color,-1.,
                c.ChVector3f(*map(float,TUBE_NORMAL)),1.60,math.pi,False)
        # First-bounce illumination from the white parts of the floor. Every
        # node radiates upward, so it cannot create a hotspot on its own plane.
        bounce_positions,bounce_strengths=floor_bounce_nodes(a.light_brightness,a.exposure)
        for p,bounce in zip(bounce_positions,bounce_strengths):
            manager.scene.AddSpotLight(c.ChVector3f(*map(float,p)),c.ChColor(*map(float,rgb*bounce)),-2.,
                c.ChVector3f(0,0,1),1.60,math.pi,False)
        for p,bounce in zip(*ceiling_bounce_nodes(a.light_brightness,a.exposure)):
            manager.scene.AddSpotLight(c.ChVector3f(*map(float,p)),c.ChColor(*map(float,rgb*bounce)),-3.,
                c.ChVector3f(0,0,-1),1.60,math.pi,False)
    sensors={}
    spatial_factor=1 if stochastic_render(a) else a.samples
    rays_per_axis=a.samples if stochastic_render(a) else 1
    response=json.loads((ROOT/'assets/reference_response.json').read_text())
    hfov=camera_metadata(a)['effective_hfov_deg']
    for name,parent,pose in [('wrist',scene.links[5],scene.wrist_optical),
                             (scene.global_camera_name,scene.world,scene.global_optical)]:
        camera=sens.ChCameraSensor(parent,a.rate,scene.frame(pose),a.width*spatial_factor,(a.width*3//4)*spatial_factor,
            math.radians(hfov),rays_per_axis,
            sens.CameraLensModelType_FOV_LENS if a.lens_model=='reference' else sens.CameraLensModelType_PINHOLE,
            a.path_traced,use_denoiser(a),sens.Integrator_PATH if stochastic_render(a) else sens.Integrator_LEGACY,2.2)
        camera.SetName('attempt-codex / '+name+' / IMX335 B ('+a.lens_model+')')
        camera.SetLag(0.); camera.SetCollectionWindow(0.)
        if spatial_factor>1:
            # One center ray per high-resolution pixel, then a fixed box average.
            # This avoids the renderer's frame-varying subpixel jitter as well.
            camera.PushFilter(sens.ChFilterImgAlias(spatial_factor))
        if not a.raw_response:
            values=response['cameras']['wrist' if name=='wrist' else 'right_45']
            # The coefficients of exposure time and frame means are zero: this
            # is a fixed monotonic response, with no per-frame autoexposure.
            camera.PushFilter(sens.ChFilterCameraExposureCorrect(0.,0.,values['contrast'],
                0.,0.,values['log_exposure_bias'],1.))
        if not a.headless:
            camera.PushFilter(sens.ChFilterVisualize(a.width,a.width*3//4,'attempt-codex | '+name.replace('_',' ')+' | IMX335 B'))
        camera.PushFilter(sens.ChFilterRGBA8Access())
        manager.AddSensor(camera); sensors[name]=camera
    return manager,sensors

class FrameRecorder:
    """Copy each newly published sensor frame, without repeating stale buffers."""
    def __init__(self,output,scene):
        from PIL import Image
        self.Image=Image; self.output=output; self.scene=scene
        self.last={}; self.records=[]

    def capture(self,sensors):
        for name,camera in sensors.items():
            data=camera.GetMostRecentRGBA8Buffer()
            if not data.HasData(): continue
            count=int(data.LaunchedCount)
            if count==self.last.get(name): continue
            self.last[name]=count
            image=np.flipud(np.array(data.GetRGBA8Data(),copy=True)).astype(np.uint8)
            target=self.output/name/f'{count:06d}.png'
            target.parent.mkdir(parents=True,exist_ok=True)
            self.Image.fromarray(image).save(target)
            scene=self.scene
            pose=scene.camera_world_pose(name)
            self.records.append(dict(camera=name,frame=count,timestamp=float(data.TimeStamp),
                file=str(target.relative_to(self.output)),world_from_optical=pose.tolist()))

    def close(self):
        self.output.mkdir(parents=True,exist_ok=True)
        a=self.scene.args
        report=dict(samples_per_pixel=a.samples**2,denoiser='on' if use_denoiser(a) else 'off',
            render_mode=render_mode(a),appearance=appearance_metadata(a),
            motion=a.motion,frames=self.records)
        (self.output/'frames.json').write_text(json.dumps(report,indent=2)+'\n')

def save_results(scene,sensors,output):
    output.mkdir(parents=True,exist_ok=True)
    poses={}
    for name,camera in sensors.items():
        data=camera.GetMostRecentRGBA8Buffer()
        if not data.HasData(): raise RuntimeError(f'{name}: no rendered sensor frame was received')
        image=np.array(data.GetRGBA8Data(),copy=True)
        # Chrono's buffers have their origin at the bottom left.
        image=np.flipud(image)
        try:
            from PIL import Image
            Image.fromarray(image.astype(np.uint8)).save(output/(name+'.png'))
        except ImportError:
            # PPM keeps export usable in a minimal PyChrono/Numpy environment.
            rgb=np.ascontiguousarray(image[:,:,:3],dtype=np.uint8)
            with (output/(name+'.ppm')).open('wb') as f:
                f.write(f'P6\n{rgb.shape[1]} {rgb.shape[0]}\n255\n'.encode()); f.write(rgb.tobytes())
        pose=scene.camera_world_pose(name)
        poses[name]=dict(world_from_optical=pose.tolist(),forward_axis='+X',up_axis='+Z',
            global_camera_config=scene.args.global_camera,
            mount_to_image_roll_deg=180 if name=='wrist' else 0,
            width=scene.args.width,height=scene.args.width*3//4,hfov_deg=camera_metadata(scene.args)['effective_hfov_deg'],
            lens_model=camera_metadata(scene.args),
            render_mode=render_mode(scene.args),
            appearance=appearance_metadata(scene.args),
            illumination_model='tube_center_point_light' if scene.args.fast else 'neewer_gc21b_inflatable_tube',
            diffuser_center_m=list(scene.diffuser_center),diffuser_size_m=list(scene.diffuser_size),
            dominant_illumination_direction=TUBE_NORMAL.tolist(),
            ambient_fill=(.02 if scene.args.path_traced else AMBIENT_FILL)*scene.args.exposure*scene.args.light_brightness/.5,
            samples_per_pixel=scene.args.samples**2,denoised=use_denoiser(scene.args),added_sensor_noise=False,
            updates=int(data.LaunchedCount))
    (output/'camera_poses.json').write_text(json.dumps(poses,indent=2)+'\n')
    print('Saved sensor frames and poses to',output)

def make_viewport(scene):
    import pychrono.irrlicht as irr
    c=scene.c
    vis=irr.ChVisualSystemIrrlicht(); vis.AttachSystem(scene.system)
    vis.SetBackgroundColor(c.ChColor(.025,.030,.024))
    vis.SetWindowSize(1100,800); vis.SetWindowTitle('attempt-codex | SO-101 white lightbox | third person')
    vis.SetCameraVertical(c.CameraVerticalDir_Z); vis.Initialize()
    vis.AddCamera(scene.vec([.34,-.64,.30]),scene.vec([0,.26,.23]))
    rgb=white_balanced_light(scene.args)
    scale=scene.args.exposure*scene.args.light_brightness/.5
    vis.AddLightDirectional(65.,40.,c.ChColor(*map(float,rgb*.75*scale)),
        c.ChColor(0,0,0),c.ChColor(*map(float,rgb*.85*scale)))
    return vis

def viewport_worker(args,assets,shared_time,shared_state,stop,ready,status):
    # Irrlicht and Sensor's GLFW teardown conflict in this installed build.
    # A spawned process gives each renderer its own native graphics lifecycle.
    import signal
    import traceback
    signal.signal(signal.SIGINT,signal.SIG_IGN)
    try:
        import pychrono.core as chrono
        scene=Scene(chrono,assets,args,simulate_jaw=False)
        with shared_state.get_lock(): scene.apply_shared_state(shared_time.value,shared_state[:])
        vis=make_viewport(scene)
        ready.set()
        while not stop.is_set() and vis.Run():
            with shared_state.get_lock(): scene.apply_shared_state(shared_time.value,shared_state[:])
            vis.BeginScene(); vis.Render(); vis.EndScene()
            time.sleep(1/30)
        if args.save_dir:
            args.save_dir.mkdir(parents=True,exist_ok=True)
            vis.WriteImageToFile(str(args.save_dir/'third_person.png'))
        del vis
        status.put(None)
    except BaseException:
        status.put(traceback.format_exc())
        ready.set()
    finally:
        stop.set()

def main():
    args=parse_args()
    assets=json.loads((ROOT/'assets/scene_assets.json').read_text())
    validate(assets)
    if args.validate: return
    try:
        import pychrono.core as chrono
    except ImportError as e:
        raise SystemExit('PyChrono is unavailable. Run ./run.sh inside the Chrono container, or ./run-docker.sh from the host.') from e
    scene=Scene(chrono,assets,args)
    controller=None;episode_writer=None
    if args.auto:
        from pick_place import PickPlaceController
        controller=PickPlaceController(scene,args)
    if args.collect_dir:
        from episode_data import EpisodeWriter
        episode_writer=EpisodeWriter(args.collect_dir.resolve(),scene,args)
        episode_writer.begin(controller)
    vis=None
    viewport=None
    if not args.headless and not args.no_sensors:
        import multiprocessing as mp
        ctx=mp.get_context('spawn')
        shared_time=ctx.Value('d',0.); stop=ctx.Event(); ready=ctx.Event(); status=ctx.Queue()
        shared_state=ctx.Array('d',scene.shared_state())
        viewport=ctx.Process(target=viewport_worker,args=(args,assets,shared_time,shared_state,stop,ready,status),daemon=True)
        viewport.start()
        if not ready.wait(30):
            viewport.terminate(); viewport.join()
            raise RuntimeError('Third-person window did not initialize within 30 seconds')
        if stop.is_set():
            viewport.join(); raise RuntimeError(status.get(timeout=2) or 'Third-person window closed during startup')
    elif not args.headless:
        vis=make_viewport(scene)
    manager=None; sensors={}
    if not args.no_sensors:
        import pychrono.sensor as sens
        manager,sensors=configure_sensors(scene,sens)
    recorder=FrameRecorder(args.record_dir.resolve(),scene) if args.record_dir and manager else None
    lighting='tube-center point approximation' if args.fast else 'fixed quadrature over the narrow tube face'
    activity='IK pick-and-place enabled.' if args.auto else ('Slow joint motion enabled.' if args.motion else 'Static inspection pose.')
    print(f'GC21B: {args.light_kelvin:g} K estimate, {args.light_brightness:.0%} brightness, '
          f'{1300*args.light_brightness:g} lux at 0.5 m nominal ({lighting}). '+activity)
    print('Green PLA cube: 3 x 3 x 3 cm, 30 g. Red tray: 5 x 5 cm, 1 cm raised edge.')
    if controller:print('Grasp physics: full CAD jaw contacts; torque-limited moving jaw; cube stays dynamic.')
    print('Close the third-person window or press Ctrl-C to exit.')
    start=time.monotonic(); t=0.; dt=min(.001,1/(args.rate*4)); next_render=0.;next_capture=0.
    try:
        while args.duration is None or t<args.duration:
            if vis and not vis.Run(): break
            if viewport and (stop.is_set() or not viewport.is_alive()): break
            if controller:
                if controller.update(t):
                    result=controller.finish_episode(t)
                    if episode_writer:episode_writer.end(result)
                    if controller.episode_index+1>=args.episodes:
                        controller.done=True;break
                    controller.start_episode(t)
                    if episode_writer:episode_writer.begin(controller)
                    next_capture=t
            else:scene.update_arm(t)
            if viewport: shared_time.value=t
            if episode_writer:
                if t>=next_capture-1e-9:
                    from episode_data import capture_synchronized
                    images,stamps,launches=capture_synchronized(scene.system,manager,sensors,t)
                    episode_writer.add(t,controller,images,stamps,launches)
                    next_capture=controller.start_time+len(episode_writer.rows)/args.rate
            elif manager: manager.Update()
            if recorder: recorder.capture(sensors)
            if vis and t>=next_render:
                vis.BeginScene(); vis.Render(); vis.EndScene(); next_render+=1/30
            step=dt
            if episode_writer:step=min(step,max(1e-10,next_capture-t))
            if controller:step=min(step,max(1e-10,controller.start_time+controller.plan.duration-t))
            scene.system.DoStepDynamics(step); t=scene.system.GetChTime()
            if controller:controller.after_step(t)
            if viewport:
                with shared_state.get_lock(): shared_state[:]=scene.shared_state()
            remaining=start+t-time.monotonic()
            if remaining>0 and not args.no_realtime: time.sleep(min(remaining,.02))
    except KeyboardInterrupt:
        pass
    finally:
        if controller and not controller.done and len(controller.results)<=controller.episode_index:
            result=controller.finish_episode(t,complete=False)
            if episode_writer:episode_writer.end(result)
        if viewport:
            shared_time.value=t
            stop.set()
            viewport.join(20)
            if viewport.is_alive():
                viewport.terminate(); viewport.join()
                raise RuntimeError('Third-person renderer did not exit within 20 seconds')
            if viewport.exitcode!=0:
                raise RuntimeError(f'Third-person renderer exited with code {viewport.exitcode}')
            failure=status.get(timeout=2)
            if failure: raise RuntimeError(failure)
    if manager:
        # Let the asynchronous render thread finish its final frame.
        time.sleep(.2)
        if recorder:
            recorder.capture(sensors)
            recorder.close()
        if args.save_dir: save_results(scene,sensors,args.save_dir.resolve())
    if vis and args.save_dir:
        args.save_dir.mkdir(parents=True,exist_ok=True)
        vis.WriteImageToFile(str(args.save_dir/'third_person.png'))
    if args.save_dir: scene.save_object_state(args.save_dir.resolve())
    if controller and (args.save_dir or args.collect_dir):
        target=(args.save_dir or args.collect_dir).resolve();target.mkdir(parents=True,exist_ok=True)
        (target/'auto_report.json').write_text(json.dumps(dict(seed=args.seed,results=controller.results),indent=2)+'\n')
    print(f'Finished at {t:.3f} s.')
    # Release GPU/window resources while all bound C++ modules are still alive.
    # Python's interpreter-shutdown finalizer order is unsafe for these backends.
    sensors.clear()
    if manager:
        del manager
    if vis:
        del vis

if __name__=='__main__':
    main()
