#!/usr/bin/env python3
"""Package real/Chrono comparison videos, source metadata and a local viewer."""
import argparse
import json
import shutil
from pathlib import Path
from PIL import Image,ImageDraw
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--fitted',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,default=ROOT/'preview/video_reference')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    baseline=json.loads((a.baseline/'replay.json').read_text())
    reports=[json.loads((root/'replay.json').read_text()) for root in a.fitted]
    result=dict(metric='Mean absolute RGB8 difference over complete 640 x 480 frames; no registration or crop',
        camera_profile=json.loads((ROOT/'assets/video_camera.json').read_text()),
        baseline=dict(episode=baseline['episode'],frames=baseline['frames'],mean_rgb_mae=baseline['mean_rgb_mae']),
        fitted=[dict(episode=r['episode'],frames=r['frames'],fps=r['fps'],mean_rgb_mae=r['mean_rgb_mae']) for r in reports],
        limitations=reports[0]['limitations']+[
            'The recorded red plate appears flatter than the retained 1 cm simulation tray rim',
            'Episode 0 was used for response fitting; episode 1 supplies an additional motion check',
            'Baseline uses the earlier nominal camera and scene; object reconstruction also depends on the selected camera pose'])
    result['episode_0_error_reduction_percent']={name:100*(1-reports[0]['mean_rgb_mae'][name]/baseline['mean_rgb_mae'][name])
                                                for name in ['external','wrist']}
    clips=[]
    for root,report in zip(a.fitted,reports):
        ep=report['episode'];entry=dict(episode=ep,frames=report['frames'],fps=report['fps'])
        for camera in ['external','wrist']:
            name=f'episode_{ep:03d}_{camera}.mp4'
            shutil.copy2(root/(camera+'_side_by_side.mp4'),a.output/name);entry[camera]=name
        shutil.copy2(root/'replay.json',a.output/f'episode_{ep:03d}_replay.json')
        clips.append(entry)
    canvas=Image.new('RGB',(1920,2*512),(24,28,33));draw=ImageDraw.Draw(canvas)
    for row,camera in enumerate(['external','wrist']):
        images=[Image.open(a.fitted[0]/f'{camera}_real_00000.png').convert('RGB'),
                Image.open(a.baseline/f'{camera}_sim_00000.png').convert('RGB'),
                Image.open(a.fitted[0]/f'{camera}_sim_00000.png').convert('RGB')]
        for col,(im,label) in enumerate(zip(images,['REAL','BEFORE','FITTED CHRONO SENSOR'])):
            canvas.paste(im,(col*640,row*512+32));draw.text((col*640+12,row*512+9),camera.upper()+' | '+label,fill='white')
    canvas.save(a.output/'before_after.jpg',quality=94)
    # Save overlays/differences for direct inspection, without registration.
    for camera in ['external','wrist']:
        real=np.array(Image.open(a.fitted[0]/f'{camera}_real_00000.png').convert('RGB')).astype(float)
        sim=np.array(Image.open(a.fitted[0]/f'{camera}_sim_00000.png').convert('RGB')).astype(float)
        Image.fromarray(np.uint8((real+sim)/2)).save(a.output/(camera+'_overlay.png'))
        Image.fromarray(np.uint8(np.clip(abs(real-sim)*3,0,255))).save(a.output/(camera+'_difference_x3.png'))
    (a.output/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    html='''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Real and Chrono Sensor video comparison</title>
<style>body{margin:30px auto;padding:0 20px;max-width:1350px;font:16px system-ui;background:#141c25;color:#edf2f8;line-height:1.5}video,img{width:100%;background:#000}h1{font-size:28px}h2{font-size:21px}select,button{font:inherit;padding:8px;margin:6px 6px 12px 0}table{border-collapse:collapse}td,th{padding:8px 22px 8px 0;text-align:left}a{color:#a2d7ff}.note{color:#b7c4d3}</style>
<h1>Recorded task: real and Chrono Sensor</h1>
<p>The real frame is on the left of each video; the synchronized sensor render is on the right.
The arm follows recorded <code>observation.state</code> through an approximate visual motor map.</p>
<label>Episode <select id="episode"></select></label><button id="toggle">Play both cameras</button>
<h2>External camera</h2><video id="external" controls playsinline preload="metadata"></video>
<h2>Wrist camera</h2><video id="wrist" controls playsinline preload="metadata"></video>
<h2>First-frame comparison</h2><p>Real / previous default / fitted sensor.</p><img src="before_after.jpg">
<h2>Measured fit</h2><p>The external floor/mat landmarks fit at 2.76 px RMS with the lens held fixed.</p>
<table><thead><tr><th>Run</th><th>External RGB MAE / 255</th><th>Wrist RGB MAE / 255</th></tr></thead><tbody>ROWS</tbody></table>
<p class="note">Errors include every RGB pixel before video compression. No image alignment or crop is applied.
Episode 0 was used for response fitting. The baseline uses the previous nominal camera and scene;
object reconstruction also depends on camera pose.</p>
<p class="note">This is kinematic visual replay. Motor coordinates, room geometry and camera response are estimates.
Cube translation is reconstructed from color centroids with fallbacks; object rotations are fixed.
The retained simulation tray has a taller rim than the recorded plate. These videos do not validate grasp physics.</p>
<p><a href="metrics.json">Fit metrics and limitations</a> · <a id="metadata">Current episode metadata</a></p>
<script>const clips=CLIPS,choice=document.querySelector('#episode'),videos=['external','wrist'].map(k=>document.getElementById(k));
for(const [i,c] of clips.entries()){const o=document.createElement('option');o.value=i;o.textContent=c.episode+' · '+c.frames+' frames at '+c.fps+' fps';choice.append(o);}
function load(){const c=clips[+choice.value];videos.forEach((v,i)=>{v.pause();v.src=c[['external','wrist'][i]];});document.querySelector('#metadata').href='episode_'+String(c.episode).padStart(3,'0')+'_replay.json';}
choice.onchange=load;document.querySelector('#toggle').onclick=()=>{const t=videos[0].currentTime;videos[1].currentTime=t;if(videos[0].paused)videos.forEach(v=>v.play());else videos.forEach(v=>v.pause());};
videos.forEach(v=>v.addEventListener('seeked',()=>{if(v.paused){const other=videos.find(x=>x!==v);if(Math.abs(other.currentTime-v.currentTime)>.1)other.currentTime=v.currentTime;}}));load();</script>'''
    rows=f'<tr><td>Previous default, episode 0 ({baseline["frames"]} frames)</td><td>{baseline["mean_rgb_mae"]["external"]:.2f}</td><td>{baseline["mean_rgb_mae"]["wrist"]:.2f}</td></tr>'
    for r in reports:
        rows+=f'<tr><td>Fitted, episode {r["episode"]} ({r["frames"]} frames)</td><td>{r["mean_rgb_mae"]["external"]:.2f}</td><td>{r["mean_rgb_mae"]["wrist"]:.2f}</td></tr>'
    (a.output/'index.html').write_text(html.replace('ROWS',rows).replace('CLIPS',json.dumps(clips)))
    print(json.dumps({k:result[k] for k in ['baseline','fitted','episode_0_error_reduction_percent']},indent=2))


if __name__=='__main__':main()
