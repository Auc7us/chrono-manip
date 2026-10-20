"""Measured fixture specs and explicit photographic estimates for the lightbox.

The lux rating is illuminance at a distance, not lumens or electrical watts.
Material reflectances, tube diameter/placement and camera response are estimates.
"""
import math
import json
from pathlib import Path
import numpy as np
from scene_config import WIDTH,DEPTH,HEIGHT,SHEET_SIZE,SHEET_CENTER

LIGHT_SPEC_URL = 'https://neewer.com/products/neewer-gc21b-21w-bi-color-handheld-led-tube-light-66606507'
LIGHT_MAX_LUX = 1300.
LIGHT_REFERENCE_DISTANCE = .5
LIGHT_NOMINAL_LENGTH = .90
LIGHT_BRIGHTNESS = .50
LIGHT_KELVIN = 4600.
WHITE_BALANCE_KELVIN = 4600.
# Tube runs across the rear ceiling edge, as seen at image right from the wall camera.
_tube=json.loads((Path(__file__).resolve().parent/'assets/reference_tube.json').read_text())
TUBE_LENGTH = _tube['installed_length_m']
TUBE_RADIUS = _tube['radius_m']
TUBE_CENTER = np.array(_tube['center_m'])
TUBE_NORMAL = np.array(_tube['emission_normal'])
TUBE_U = np.array(_tube['axis'])
_y=np.array([0.,1.,0.]);_y-=TUBE_U*(TUBE_U@_y);_y/=np.linalg.norm(_y)
TUBE_FRAME=np.column_stack([TUBE_U,_y,np.cross(TUBE_U,_y)])
TUBE_WHITE_COSINE=_tube['white_sector_cosine']
TUBE_V = np.cross(TUBE_NORMAL, TUBE_U)
EMITTER_CENTER = TUBE_CENTER + TUBE_NORMAL*(TUBE_RADIUS+.002)
EMITTER_SIZE = (TUBE_LENGTH-.14, 2*TUBE_RADIUS*math.sqrt(1-TUBE_WHITE_COSINE**2))
# This converts illuminance to the renderer's relative display scale; it is not
# a measurement of the IMX335's shutter, gain, spectral response or lens T-stop.
CAMERA_LIGHT_SCALE = .0010
AMBIENT_FILL = .055

COLORS = dict(cream=(.83,.81,.72), black=(.012,.014,.013), metal=(.38,.39,.40),
              electronics=(.06,.13,.09), white=(.80,.82,.79), red=(.62,.012,.025),
              cable_white=(.81,.82,.77), pcb=(.012,.013,.012), glass=(.018,.04,.045),
              diffuser=(.92,.94,.91), gold=(.63,.43,.12), pla=(.025,.43,.085),
              floor_sheet=(.030,.036,.030), panel_edge=(.30,.29,.25),
              room_wall=(.20,.28,.14), room_floor=(.035,.038,.028),
              room_dark=(.008,.011,.008), room_book=(.038,.055,.049),
              room_blue=(.018,.037,.080),room_green=(.025,.065,.040),
              room_rust=(.17,.045,.020),room_paper=(.19,.20,.15),
              tape=(.73,.74,.66), screw_recess=(.006,.007,.006),rack_metal=(.012,.014,.012),
              polished_metal=(.62,.64,.68))


def quadrature(nx=16, ny=3):
    """Fixed integration points on the tube's inward/downward luminous face."""
    u = ((np.arange(nx)+.5)/nx-.5)*EMITTER_SIZE[0]
    v = ((np.arange(ny)+.5)/ny-.5)*EMITTER_SIZE[1]
    return np.array([EMITTER_CENTER+x*TUBE_U+y*TUBE_V for x in u for y in v])


def node_strength(brightness):
    """Normalize a finite emitter to the rated axial illuminance at 0.5 m.

    Chrono's legacy point shader approximates the emitter angular response;
    the absolute camera photometry remains uncalibrated.
    """
    target=EMITTER_CENTER+TUBE_NORMAL*LIGHT_REFERENCE_DISTANCE
    rays=target-quadrature()
    distances=np.linalg.norm(rays,axis=1)
    cosines=(rays@TUBE_NORMAL)/distances
    angles=np.arccos(np.clip(cosines,0,1))
    emission=np.clip((math.pi/2-angles)/(math.pi/2-.80),0,1)**2
    geometric_sum=np.sum(emission*cosines**2/distances**2)
    return LIGHT_MAX_LUX*brightness/geometric_sum*CAMERA_LIGHT_SCALE


def floor_bounce_nodes(brightness,exposure=1.,nx=8,ny=6):
    """Deterministic first-bounce radiosity from the exposed white floor.

    Black mat cells are excluded. Each white cell reflects its incident tube
    illumination with rho/pi and emits upward. This replaces the strong,
    view-dependent ambient term with a spatial white-panel bounce estimate.
    """
    positions=[];strengths=[];area=WIDTH*DEPTH/(nx*ny)
    for x in ((np.arange(nx)+.5)/nx-.5)*WIDTH:
        for y in (np.arange(ny)+.5)/ny*DEPTH:
            if abs(x)<SHEET_SIZE[0]/2 and abs(y-SHEET_CENTER[1])<SHEET_SIZE[1]/2:continue
            p=np.array([x,y,.004]);rays=quadrature()-p
            distances=np.linalg.norm(rays,axis=1);direction=rays/distances[:,None]
            receiver=np.maximum(direction[:,2],0)
            angle=np.arccos(np.clip((-direction)@TUBE_NORMAL,0,1))
            emission=np.clip((math.pi/2-angle)/(math.pi/2-.80),0,1)**2
            incoming=np.sum(emission*receiver**2/distances**2)*node_strength(brightness)*exposure
            strengths.append(incoming*area/math.pi*np.array(COLORS['white']))
            positions.append(p)
    return np.array(positions),np.array(strengths)


def ceiling_bounce_nodes(brightness,exposure=1.,nx=6,ny=4):
    """Second diffuse return from floor to ceiling, then down into the box."""
    floor,source=floor_bounce_nodes(brightness,exposure)
    positions=[];strengths=[];depth=DEPTH-.16;area=WIDTH*depth/(nx*ny)
    for x in ((np.arange(nx)+.5)/nx-.5)*WIDTH:
        for y in (np.arange(ny)+.5)/ny*depth:
            p=np.array([x,y,HEIGHT-.004]);rays=floor-p
            distances=np.linalg.norm(rays,axis=1);directions=rays/distances[:,None]
            cosine=np.clip(-directions[:,2],0,1);angles=np.arccos(cosine)
            emission=np.clip((math.pi/2-angles)/(math.pi/2-.80),0,1)**2
            incoming=np.sum(source*(emission*cosine**2/distances**2)[:,None],axis=0)
            positions.append(p);strengths.append(incoming*area/math.pi*np.array(COLORS['white']))
    return np.array(positions),np.array(strengths)


def appearance_metadata(args):
    from reference_camera import camera_metadata
    response=json.loads((Path(__file__).resolve().parent/'assets/reference_response.json').read_text())
    return dict(profile='real_lightbox_2026_10_19_fitted',
        interior_dimensions_m=[WIDTH,DEPTH,HEIGHT],
        light=dict(model='NEEWER GC21B',specification_url=LIGHT_SPEC_URL,
            maximum_electrical_power_w=21,cri_nominal=95,
            rated_illuminance_lux=LIGHT_MAX_LUX,rating_distance_m=LIGHT_REFERENCE_DISTANCE,
            brightness_fraction=args.light_brightness,
            estimated_reference_illuminance_lux=LIGHT_MAX_LUX*args.light_brightness,
            cct_k=args.light_kelvin,cct_is_estimate=True,nominal_inflated_length_m=LIGHT_NOMINAL_LENGTH,
            installed_length_m=TUBE_LENGTH,estimated_radius_m=TUBE_RADIUS,
            center_m=TUBE_CENTER.tolist(),emitter_center_m=EMITTER_CENTER.tolist(),
            axis=TUBE_U.tolist(),backing_fit=_tube,
            emission_normal=TUBE_NORMAL.tolist(),emitter_size_m=list(EMITTER_SIZE),
            integration_nodes=len(quadrature())),
        camera=dict(white_balance_k=args.white_balance_kelvin,exposure_scale=args.exposure,
            response_calibrated=False,projection=camera_metadata(args),
            wrist_optical_fit_applied=args.lens_model=='reference',
            fixed_response=None if args.raw_response else response,
            wrist_optical_fit=json.loads((Path(__file__).resolve().parent/'assets/reference_wrist.json').read_text())),
        indirect_light=dict(model='deterministic floor/ceiling diffuse return; finite-cell near-field approximation',
            enabled=not(args.fast or args.path_traced or args.stochastic_area),
            floor_cells=len(floor_bounce_nodes(args.light_brightness)[0]),
            ceiling_cells=len(ceiling_bounce_nodes(args.light_brightness)[0]),
            ambient_residual=AMBIENT_FILL),
        materials=dict(linear_rgb_reflectance={k:list(v) for k,v in COLORS.items()},
            metal=dict(roughness=.14,metallic=.96,specular_rgb=[.72,.74,.77]),
            adapter_screw_heads=dict(roughness=.095,metallic=.98,radius_m=.0024,workflow='metallic',
                shape='domed cap with curved bevel and recessed Phillips opening'),
            wrist_camera_mount='black PLA',surface_detail='deterministic procedural estimates',
            mat_collision='flat 1 mm sheet; submillimeter visual creases only'),
        panels=dict(estimated_wall_gap_m=[.0016,.0030],estimated_bow_m=.0013,
                    front_floor_curl_m=.0014,open_seams=True),
        estimates=['tube diameter and ceiling placement','panel and PLA reflectance/roughness',
            'white balance and display exposure','room geometry beyond open front'])
