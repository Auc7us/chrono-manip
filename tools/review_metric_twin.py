#!/usr/bin/env python3
"""Package metric calibration, physical replays and every validation outcome."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fit',type=Path,required=True);p.add_argument('--validation',type=Path,required=True)
    p.add_argument('--videos',type=Path,nargs='+',required=True)
    p.add_argument('--renders',type=Path,required=True)
    p.add_argument('--baseline-validation',type=Path)
    p.add_argument('--clearance',type=Path)
    p.add_argument('--baseline-fit',type=Path)
    p.add_argument('--baseline-renders',type=Path)
    p.add_argument('--baseline-clearance',type=Path)
    p.add_argument('--output',type=Path,default=ROOT/'preview/metric_reference')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    fit=json.loads(a.fit.read_text());results=json.loads((a.validation/'summary.json').read_text());clips=[]
    for root in a.videos:
        data=json.loads((root/'physics_replay.json').read_text());ep=data['episode']
        entry=dict(episode=ep,frames=data['frames'],fps=data['fps'])
        for camera in ['external','wrist']:
            name=f'episode_{ep:03d}_{camera}.mp4';shutil.copy2(root/(camera+'_physical.mp4'),a.output/name);entry[camera]=name
        shutil.copy2(root/'physics_replay.json',a.output/f'episode_{ep:03d}_physics.json');clips.append(entry)
    for row in results:
        source=a.validation/f'episode_{row["episode"]:03d}/physics_replay.json'
        if source.exists():shutil.copy2(source,a.output/f'validation_{row["episode"]:03d}.json')
    report=dict(calibration=fit,encoder_units=json.loads((ROOT/'assets/encoder_units.json').read_text()),
        external_camera=json.loads((ROOT/'assets/video_camera.json').read_text()),physical_validation=results,
        gripper_response=json.loads((ROOT/'assets/recorded_jaw_servo.json').read_text()),
        limitations=['Image-derived calibration and rigid CAD geometry; physical servo/friction/compliance parameters are unmeasured',
            'The first five arm joints remain prescribed from measured states; only the gripper/cube use contact dynamics',
            'Final reference position is reconstructed from both images on an assumed plate plane, not independently surveyed',
            'All contact outcomes and force spikes are retained; matching images does not guarantee matching dynamics'])
    data=np.load(ROOT/'output/video_fit/reference_trajectory.npz');states=data['observation_state'];actions=data['action'];response_check={}
    for label,directory in [('current',a.validation),('previous_model',a.baseline_validation)]:
        if directory is None:continue
        errors=[];loaded=[]
        for file in sorted(directory.glob('episode_*/physics_replay.json')):
            physical=json.loads(file.read_text())
            if physical['episode']<8:continue
            mapping=physical['joint_map']
            for row in physical['records']:
                state=states[row['dataset_index'],5]
                measured=state*mapping['scale'][5]+mapping['offset_deg'][5]
                if actions[row['dataset_index'],5]<1 and 5<state<20:loaded.append(row['jaw_actual_deg']-measured)
                if state>25:
                    errors.append(row['jaw_actual_deg']-measured)
        response_check[label]=dict(samples=len(errors),held_out_open_gap_position_rms_deg=float(np.sqrt(np.mean(np.array(errors)**2))),
            held_out_loaded_samples=len(loaded),held_out_loaded_position_rms_deg=float(np.sqrt(np.mean(np.square(loaded)))))
    report['measured_gripper_response_check']=response_check
    def aggregate(rows):
        rows=[r for r in rows if 'error' not in r]
        return dict(episodes=len(rows),cube_centers_on_plate=sum(r['cube_inside_plate'] for r in rows),
            median_endpoint_xy_error_mm=float(np.median([r['final_xy_error_m'] for r in rows])*1000))
    report['aggregate_validation']={'current':aggregate(results)}
    if a.baseline_validation:report['aggregate_validation']['previous_model']=aggregate(json.loads((a.baseline_validation/'summary.json').read_text()))
    hardware=ROOT/'assets/recorded_motor_calibration.json'
    if hardware.exists():report['hardware_motor_calibration']=json.loads(hardware.read_text())
    contact=ROOT/'assets/recorded_contact_response.json'
    if contact.exists():report['effective_contact_response']=json.loads(contact.read_text())
    if a.clearance:report['recorded_jaw_floor_clearance']=json.loads(a.clearance.read_text())
    if a.baseline_clearance:report['previous_recorded_jaw_floor_clearance']=json.loads(a.baseline_clearance.read_text())
    if a.baseline_fit:report['previous_calibration']=json.loads(a.baseline_fit.read_text())
    (a.output/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    # Before/after uses the same original source frame, with no image registration.
    before=ROOT/'output/video_fit/fitted_episode_000'
    canvas=Image.new('RGB',(1920,1024),(24,28,33));draw=ImageDraw.Draw(canvas)
    for row,(camera,sensor) in enumerate([('external','right_45'),('wrist','wrist')]):
        previous=a.baseline_renders/(sensor+'.png') if a.baseline_renders else before/f'{camera}_sim_00000.png'
        files=[ROOT/f'output/video_fit/real_{camera}_000.png',previous,a.renders/(sensor+'.png')]
        for col,(file,label) in enumerate(zip(files,['REAL','PREVIOUS FIT','CORRECTED DEFAULT'])):
            canvas.paste(Image.open(file).convert('RGB'),(col*640,row*512+32));draw.text((col*640+12,row*512+9),camera.upper()+' | '+label,fill='white')
    canvas.save(a.output/'before_after.jpg',quality=94)
    rows=''
    for row in results:
        if 'error' in row:rows+=f'<tr><td>{row["episode"]}</td><td colspan="4">Run failed; see log</td></tr>';continue
        outcome='Within 5 mm of reference' if row['final_xy_error_m']<=.005 else 'Endpoint mismatch'
        rows+=f'<tr><td>{row["episode"]}</td><td>{outcome}</td><td>{row["final_xy_error_m"]*1000:.1f}</td><td>{row["maximum_penetration_m"]*1000:.3f}</td><td>{row["maximum_contact_force_n"]:.1f}</td></tr>'
    held=fit['after']['held_out'];before_held=fit['before']['held_out']
    if a.baseline_fit:before_held=report['previous_calibration']['after']['held_out']
    html='''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Metric twin and physical contact replay</title><style>body{max-width:1400px;margin:28px auto;padding:0 20px;background:#141d26;color:#edf2f8;font:16px system-ui;line-height:1.5}h1{font-size:28px}h2{font-size:21px}img,video{width:100%;background:black}table{border-collapse:collapse}th,td{padding:9px 20px 9px 0;text-align:left;border-bottom:1px solid #35414f}button,select{padding:8px;font:inherit;margin:8px}a{color:#a3d5ff}.note{color:#b8c7d5}</style>
<h1>Metric calibration and physical contact replay</h1>
<p>The unsupported white support slab has been removed. Robot mounting, motor coordinates, separate camera optics,
initial object poses, the shallow red plate and fixed material/lighting estimates now share one scene.</p>
<p>The supplied follower and leader calibration files are checked against every saved encoder value.
Commands pass through follower tick quantization; firmware homing offsets are applied once.
The files do not specify surveyed CAD joint zeros. Training poses also constrain the full CAD jaws above the mat.</p>
<h2>Default views: real / previous / corrected</h2><img src="before_after.jpg">
<h2>Actual contact replay</h2><p>Real video is on the left. The cube on the right is initialized once and stays dynamic and collision-enabled.
Later image measurements never reposition it. The first five arm joints follow observed states and the jaw uses recorded commands.</p>
<select id="episode"></select><button id="play">Play both cameras</button>
<h2>External</h2><video id="external" controls playsinline preload="metadata"></video>
<h2>Wrist</h2><video id="wrist" controls playsinline preload="metadata"></video>
<h2>Calibration checks</h2><p>165 paired views; episodes 0–7 train the fit and episodes 8–9 are held out.
Masked held-out mat-edge residual: BEFORE px → AFTER px. These residuals include the fit's visibility weights;
they are not a measured millimeter contact tolerance. Encoder-derived angular scales are fixed during calibration.</p>
<p>CLEARANCE</p>
<h2>Every physical replay outcome</h2><p>Endpoint error compares the simulated cube center to its reconstructed real final position.
An on-plate center does not establish a fully contained footprint. High force spikes and penetration expose remaining contact-model error.</p>
<p>AGGREGATE</p>
<table><thead><tr><th>Episode</th><th>Placement</th><th>Endpoint XY error, mm</th><th>Peak penetration, mm</th><th>Peak reported force, N</th></tr></thead><tbody>ROWS</tbody></table>
<p class="note">The gripper uses recorded unloaded response timing. Effective contact compliance is selected by loaded-jaw tracking on training episodes;
it is a surrogate for omitted arm/jaw flexibility. Material stiffness, loaded servo impedance and friction remain independently unmeasured.
The report retains unsuccessful cases. This is an improvement in alignment and diagnosis, not a claim of a fully calibrated physical twin.</p>
<p><a href="metrics.json">Complete fit, motor-unit inference and validation measurements</a></p>
<script>const clips=CLIPS,select=document.querySelector('#episode'),videos=['external','wrist'].map(k=>document.getElementById(k));
for(const [i,c] of clips.entries()){const option=document.createElement('option');option.value=i;option.textContent='Episode '+c.episode+' · '+c.frames+' frames';select.append(option);}
function load(){const c=clips[+select.value];videos.forEach((v,i)=>{v.pause();v.src=c[['external','wrist'][i]];});}select.onchange=load;
document.querySelector('#play').onclick=()=>{const start=videos[0].paused;videos[1].currentTime=videos[0].currentTime;videos.forEach(v=>start?v.play():v.pause());};load();</script>'''
    clearance_note=''
    if a.clearance:
        def deepest(data):return max(e[name]['penetration_below_mat_m'] for e in data['episodes'] for name in ['fixed_jaw','moving_jaw'])*1000
        clearance_note=f'Maximum recorded-pose jaw intersection with the mat: {deepest(report["recorded_jaw_floor_clearance"]):.2f} mm.'
        if a.baseline_clearance:clearance_note+=f' Previous fit: {deepest(report["previous_recorded_jaw_floor_clearance"]):.2f} mm. The revised fit accepts a small image residual increase to reduce this physical inconsistency.'
    current=report['aggregate_validation']['current']
    aggregate_note=f'Cube centers on the plate: {current["cube_centers_on_plate"]}/{current["episodes"]}. Median reference endpoint error: {current["median_endpoint_xy_error_mm"]:.2f} mm.'
    if 'previous_model' in report['aggregate_validation']:
        previous=report['aggregate_validation']['previous_model'];aggregate_note+=f' Previous model: {previous["cube_centers_on_plate"]}/{previous["episodes"]}, {previous["median_endpoint_xy_error_mm"]:.2f} mm.'
    aggregate_note+=f' Held-out loaded-jaw tracking RMS: {response_check["current"]["held_out_loaded_position_rms_deg"]:.2f}°.'
    html=html.replace('BEFORE',f'{before_held["mat_boundary_rms_px"]:.2f}').replace('AFTER',f'{held["mat_boundary_rms_px"]:.2f}').replace('ROWS',rows).replace('CLIPS',json.dumps(clips)).replace('CLEARANCE',clearance_note).replace('AGGREGATE',aggregate_note)
    (a.output/'index.html').write_text(html)
    print('Packaged metric/contact review:',a.output)


if __name__=='__main__':main()
