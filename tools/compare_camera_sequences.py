"""Compare recorded camera sequences and build a local frame-by-frame viewer.

Usage: python3 tools/compare_camera_sequences.py BEFORE AFTER --output REVIEW
Requires NumPy and Pillow. This examines real rendered frames, not an animation
made by repeating a single screenshot.
"""
import argparse
import json
import os
from pathlib import Path
import numpy as np
from PIL import Image

def read_run(root):
    meta=json.loads((root/'frames.json').read_text())
    result={}
    for camera in dict.fromkeys(r['camera'] for r in meta['frames']):
        records=[r for r in meta['frames'] if r['camera']==camera and r['timestamp']>=.4][:12]
        if len(records)<2: raise RuntimeError(f'{root}: at least two settled {camera} frames are required')
        images=np.stack([np.array(Image.open(root/r['file']).convert('RGB')) for r in records])
        delta=np.abs(np.diff(images.astype(float),axis=0))
        poses=np.array([r['world_from_optical'] for r in records])
        result[camera]=dict(records=records,metrics=dict(frames=len(records),
            camera_pose_max_delta=float(np.max(abs(poses-poses[0]))),
            mean_frame_difference_rgb8=float(delta.mean()),
            changed_pixel_fraction=float(np.mean(delta.max(3)>0))))
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('before',type=Path); p.add_argument('after',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=True)
    runs=[read_run(a.before),read_run(a.after)]
    if not runs[0] or set(runs[0])!=set(runs[1]):
        raise ValueError('Compare recordings with the same camera configuration')
    metrics={name:{cam:data['metrics'] for cam,data in run.items()} for name,run in zip(['before','after'],runs)}
    (a.output/'comparison.json').write_text(json.dumps(metrics,indent=2)+'\n')
    viewer={}
    for cam in runs[0]:
        count=min(len(run[cam]['records']) for run in runs)
        viewer[cam]=[]
        for i in range(count):
            row=[]
            for root,run in zip([a.before,a.after],runs):
                record=run[cam]['records'][i]
                row.append(dict(src=Path(os.path.relpath(root/record['file'],a.output)).as_posix(),
                    label=f"Frame {record['frame']} · {record['timestamp']:.3f} s"))
            viewer[cam].append(row)
    html='''<!doctype html><meta charset="utf-8"><title>Camera sequence comparison</title>
<style>body{font:16px system-ui;background:#18212c;color:#eef3fa;margin:28px}button,select,input{font:inherit;margin:8px}main{display:grid;grid-template-columns:1fr 1fr;gap:16px}img{width:100%;image-rendering:auto}label{display:block}h2{font-size:20px}</style>
<h1>Consecutive camera frames</h1><p>Separate rendered frames from the two supplied recordings.</p>
<select id="camera"></select>
<button id="play">Pause</button><input id="frame" type="range" min="0" value="0"><span id="position"></span>
<main><section><h2>Before</h2><img id="before"><label id="beforeLabel"></label></section><section><h2>After</h2><img id="after"><label id="afterLabel"></label></section></main>
<p>Camera motion and frame differences are measured in comparison.json.</p>
<script>const frames=FRAMES;const camera=document.querySelector('#camera'),slider=document.querySelector('#frame');let playing=true;
for(const key of Object.keys(frames)){const option=document.createElement('option');option.value=key;option.textContent=key.replaceAll('_',' ')+' camera';camera.append(option);}
function show(){const rows=frames[camera.value];slider.max=rows.length-1;const i=Math.min(+slider.value,rows.length-1);slider.value=i;for(const [j,key] of ['before','after'].entries()){document.getElementById(key).src=rows[i][j].src;document.getElementById(key+'Label').textContent=rows[i][j].label;}document.querySelector('#position').textContent=(i+1)+' / '+rows.length;}
slider.oninput=show;camera.onchange=()=>{slider.value=0;show();};document.querySelector('#play').onclick=e=>{playing=!playing;e.target.textContent=playing?'Pause':'Play';};setInterval(()=>{if(playing){slider.value=(+slider.value+1)%frames[camera.value].length;show();}},1000/15);show();</script>'''
    (a.output/'index.html').write_text(html.replace('FRAMES',json.dumps(viewer)))
    print(json.dumps(metrics,indent=2)); print('Viewer:',a.output/'index.html')

if __name__=='__main__': main()
