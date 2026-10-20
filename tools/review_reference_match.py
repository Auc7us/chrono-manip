#!/usr/bin/env python3
"""Write a portable photograph/render comparison with unmasked RGB errors."""
from pathlib import Path
import argparse
import hashlib
import html
import json
import shutil
import sys
import numpy as np
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from reference_camera import PHOTO_FIT_PROFILE,load_profile
from scene_config import WIDTH


def rgb(path):return np.array(Image.open(path).convert('RGB'))


def metrics(reference,render):
    if reference.shape!=render.shape:raise ValueError('Reference/render dimensions must match; comparison does not resize or warp images')
    delta=render.astype(float)-reference.astype(float)
    return dict(rgb_mae_8bit=float(np.abs(delta).mean()),rgb_rmse_8bit=float(np.sqrt(np.mean(delta**2))))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--renders',type=Path,required=True)
    parser.add_argument('--camera-profile',type=Path,default=PHOTO_FIT_PROFILE,
                        help='profile used to produce these renders; defaults to optional photo-fitted pose')
    parser.add_argument('--baseline',type=Path,default=ROOT/'output/appearance_tube_v4')
    parser.add_argument('--references',type=Path,default=ROOT/'output/reference_inputs')
    parser.add_argument('--sequence',type=Path,help='optional recorded stationary sequence to measure stability')
    parser.add_argument('--output',type=Path,default=ROOT/'preview/reference_match')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    sections=[];summary={};contact=Image.new('RGB',(1280,1024),'#161c20');draw=ImageDraw.Draw(contact)
    for row,(camera,reference_name) in enumerate([('right_45','global'),('wrist','wrist')]):
        photo_path=args.references/('reference_'+reference_name+'.jpg');render_path=args.renders/(camera+'.png')
        photo=rgb(photo_path);render=rgb(render_path);result=metrics(photo,render)
        result['reference_sha256']=hashlib.sha256(photo_path.read_bytes()).hexdigest()
        baseline_path=args.baseline/(camera+'.png')
        if baseline_path.exists():
            before=metrics(photo,rgb(baseline_path));result['baseline']=before
            result['mae_reduction_percent']=100*(1-result['rgb_mae_8bit']/before['rgb_mae_8bit'])
            shutil.copy2(baseline_path,args.output/(camera+'_before.png'))
        summary[camera]=result
        shutil.copy2(photo_path,args.output/(camera+'_real.jpg'));shutil.copy2(render_path,args.output/(camera+'_sim.png'))
        difference=np.mean(np.abs(render.astype(float)-photo),axis=2)
        heat=np.stack([np.minimum(difference*4,255),np.minimum(difference*1.1,200),np.minimum(difference*.15,80)],axis=2)
        Image.fromarray(heat.astype(np.uint8)).save(args.output/(camera+'_difference.png'))
        Image.fromarray(((render.astype(float)+photo)/2).astype(np.uint8)).save(args.output/(camera+'_overlay.png'))
        contact.paste(Image.fromarray(photo),(0,row*512+32));contact.paste(Image.fromarray(render),(640,row*512+32))
        draw.text((12,row*512+10),camera+' — real photo',fill='white')
        draw.text((652,row*512+10),camera+' — simulation',fill='white')
        before=result.get('baseline',{}).get('rgb_mae_8bit')
        description=(f"MAE {before:.2f} → {result['rgb_mae_8bit']:.2f}/255; {result['mae_reduction_percent']:.1f}% lower" if before is not None
                     else f"MAE {result['rgb_mae_8bit']:.2f}/255")
        sections.append(f'''<section><h2>{html.escape(camera)} <small>{description}</small></h2>
<div class="compare"><img src="{camera}_real.jpg" alt="Real photograph"><img id="{camera}" src="{camera}_sim.png" alt="Simulated camera frame"></div>
<div class="controls"><button onclick="show('{camera}','real')">Photo</button><button onclick="show('{camera}','sim')">Simulation</button><button onclick="show('{camera}','overlay')">50% overlay</button><button onclick="show('{camera}','difference')">Difference ×4</button>
<label>Opacity <input type="range" min="0" max="1" step=".01" value="1" oninput="document.getElementById('{camera}').style.opacity=this.value"></label></div></section>''')
    profile=load_profile(args.camera_profile);position=profile['optical_position_m'];angles=profile['yaw_pitch_roll_deg']
    summary['fixed_camera']=profile
    summary['camera_profile_file']=str(args.camera_profile)
    summary['comparison']='Full 640 x 480 RGB frames; no registration, crop, warp or exclusion mask in these metrics'
    summary['limitations']='Single-image geometric estimate with nominal box/mat dimensions; not a surveyed physical calibration'
    if args.sequence:
        sequence=json.loads((args.sequence/'frames.json').read_text());stability={}
        for name in ['right_45','wrist']:
            records=[r for r in sequence['frames'] if r['camera']==name]
            images=[rgb(args.sequence/r['file']).astype(int) for r in records]
            settled=images[3:]
            stability[name]=dict(recorded_frames=len(images),settling_frames_skipped=3,
                settled_frames=len(settled),maximum_settled_pixel_delta=max((np.abs(a-settled[-1]).max().item() for a in settled),default=0))
        summary['stationary_stability']=stability
    (args.output/'metrics.json').write_text(json.dumps(summary,indent=2)+'\n')
    contact.save(args.output/'comparison.png')
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><title>Real setup / simulation comparison</title>
<meta name="viewport" content="width=device-width, initial-scale=1"><style>
body{font:16px system-ui;background:#10171c;color:#edf3f5;max-width:1350px;margin:30px auto;padding:0 20px}h1{font-size:27px}h2{font-size:20px}small{font-size:15px;font-weight:400;color:#aec8d2}p{max-width:1000px;line-height:1.5}main{display:grid;grid-template-columns:1fr 1fr;gap:24px}.compare{position:relative;aspect-ratio:4/3;background:black}.compare img{position:absolute;inset:0;width:100%;height:100%}.controls{display:flex;gap:8px;flex-wrap:wrap;margin:12px 0}button{background:#263c46;color:white;border:1px solid #526c78;border-radius:5px;padding:8px;cursor:pointer}label{display:flex;gap:8px;align-items:center}a{color:#91d6eb}code{background:#23323a;padding:2px 5px}@media(max-width:850px){main{grid-template-columns:1fr}}
</style><h1>Real setup / simulation comparison</h1>
<p>Right-45 is the default installed-camera view, with an approximate centered side-top mount and 40 mm optical inset. The optional top-down view remains available. This comparison uses the camera profile stated below; these are actual Chrono sensor frames with the fitted wide-angle projection and fixed camera response.</p>'''
    page+=f'''<p>Comparison profile: <code>{html.escape(args.camera_profile.name)}</code>. The fixed lens center is estimated at <b>{100*(WIDTH/2-position[0]):.2f} cm inward from the right wall, {100*position[1]:.2f} cm from the open front, {100*position[2]:.2f} cm above the floor</b>. Yaw {angles[0]:.2f}°, downward pitch {angles[1]:.2f}°, optical roll {angles[2]:.2f}°.</p>'''
    if profile.get('landmark_rms_px') is not None:
        page+=f'''<p>Optional photograph-fitted pose: eight floor/mat landmarks, <b>{profile['landmark_rms_px']:.2f} px RMS</b>. This pose is preserved separately from the nominal default mount.</p>'''
    page+='<main>'+''.join(sections)+'</main>'
    page+='''<p>Errors use the complete 640 × 480 RGB frame, including the room, arm, light and image noise. No registration, crop, warp or exclusion mask is applied when measuring error. The image-based pose and optical fit are estimates constrained by the nominal 35 × 24 × 20 inch box and 18 × 12 inch sheet.</p>
<p>The scene uses 3D geometry and procedural materials: bowed panels with open gaps, exposed board edges, an inflatable black-backed tube, floor/ceiling light return, reflective screw heads, bookshelf, wire cart, outlet and chair. Reference photos serve as fitting targets.</p>
<p><a href="metrics.json">Parameters and full error measurements</a> · <a href="comparison.png">Two-camera contact sheet</a></p>
<script>function show(id,mode){let img=document.getElementById(id);img.src=id+(mode==='difference'?'_difference.png':'_sim.png');img.style.opacity=mode==='real'?0:mode==='overlay'?.5:1}</script></html>'''
    (args.output/'index.html').write_text(page)
    print(json.dumps({k:v for k,v in summary.items() if k in ['right_45','wrist']},indent=2))


if __name__=='__main__':main()
