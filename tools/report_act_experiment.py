#!/usr/bin/env python3
"""Build a local, self-contained ACT result page linking saved rollout videos."""
import argparse
from html import escape
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path)
    p.add_argument('--training-log',type=Path,default=ROOT/'output/act_setup/train_full.log')
    p.add_argument('--starting-step',type=int,default=0);a=p.parse_args()
    report=json.loads((a.run/'validation.json').read_text());split=json.loads((ROOT/'output/act_data/split/split.json').read_text())
    parts=['<!doctype html><meta charset="utf-8"><title>SO101 ACT evaluation</title>',
        '<style>body{font:17px system-ui;max-width:1100px;margin:40px auto;padding:0 20px;color:#192330;background:#f6f8fb}h1,h2{color:#152c44}table{border-collapse:collapse;width:100%;margin:20px 0}td,th{padding:10px;border-bottom:1px solid #ccd5df;text-align:left}video{width:100%;background:#000}section{background:white;padding:20px;margin:20px 0;border-radius:10px}small{color:#536174}a{color:#175c9a}</style>',
        '<h1>SO101 ACT: green cube into raised red tray</h1>',
        f'<p>Local RTX 4080 training. {len(split["train_indices"])} successful training episodes; {len(split["validation_indices"])} held-out episodes. Repeated tray layouts stay in one partition.</p>',
        '<p>Inputs: wrist and external RGB images plus six joint positions. Outputs: SO101 absolute joint targets. The following rollouts use learned actions with physical cube and jaw contacts.</p>']
    log=a.training_log;points=[]
    if log.exists():
        for line in log.read_text().splitlines():
            match=re.search(r'step:([\d,.]+)([KMB]?).*?loss:([\d.]+)',line)
            if match:
                multiplier={'':1,'K':1000,'M':1000000,'B':1000000000}[match[2]]
                points.append((int(float(match[1].replace(',',''))*multiplier),float(match[3])))
    if points:
        steps=points[-1][0];peak=max(y for _,y in points)
        xy=' '.join(f'{55+900*x/steps:.1f},{230-190*y/peak:.1f}' for x,y in points)
        lineage=f' after {a.starting_step:,} prior updates ({steps+a.starting_step:,} total)' if a.starting_step else ''
        parts.append(f'<section><h2>Training</h2><p>{steps:,} optimizer steps{lineage}. Final logged composite loss: {points[-1][1]:.3f}. This loss includes the training VAE term and does not measure physical task success.</p><svg viewBox="0 0 1000 270" role="img" aria-label="Training loss versus optimizer steps"><path d="M55 25V230H960" fill="none" stroke="#888"/><polyline points="{xy}" fill="none" stroke="#2463ad" stroke-width="2"/><text x="55" y="255">0</text><text x="850" y="255">{steps:,} steps</text><text x="8" y="35">{peak:.1f}</text></svg></section>')
    parts.append(f'<section><h2>Held-out prediction error</h2><p>{report["observations"]} observations, using the inference prior latent and saved training normalization.</p><table><tr><th>Joint</th><th>First-action MAE</th><th>Chunk MAE</th><th>Chunk RMSE</th></tr>')
    for i,name in enumerate(['Shoulder pan','Shoulder lift','Elbow flex','Wrist flex','Wrist roll','Gripper']):
        unit='°' if i<5 else ' percentage points'
        parts.append(f'<tr><td>{name}</td><td>{report["first_action_mae_per_joint"][i]:.2f}{unit}</td><td>{report["mean_absolute_error_per_joint"][i]:.2f}{unit}</td><td>{report["root_mean_square_error_per_joint"][i]:.2f}{unit}</td></tr>')
    parts.append('</table></section>')
    for key,title in [('held_out','Held-out recorded layouts'),('fresh','Fresh seeded layouts')]:
        folder=a.run/('rollouts_'+key);results=json.loads((folder/'results.json').read_text())
        n=len(results);successes=sum(r['success'] for r in results)
        placements=sum(r['placement_success'] for r in results)
        parts.append(f'<section><h2>{title}</h2><p><b>{placements}/{n} settled placements; {successes}/{n} also meet contact-quality limits.</b> Small initial test set; this is not a real-hardware success rate.</p><table><tr><th>Episode</th><th>Grasp</th><th>Lift</th><th>Release</th><th>In tray</th><th>Contact quality</th><th>Success</th></tr>')
        for r in results:
            cells=''.join(f'<td>{"Yes" if r[k] else "No"}</td>' for k in ['grasped','lifted','released','cube_in_tray','contact_quality_ok','success'])
            parts.append(f'<tr><td>{r["episode"]}</td>{cells}</tr>')
        horizon=results[0].get('maximum_control_horizon_s',10);settle=results[0].get('settle_s',1)
        parts.append(f'</table><p>Success requires grasp, lift, release, a settled cube entirely inside the tray, and jaw/environment penetration below 0.5 mm. Each test allows {horizon:g} seconds of control plus {settle:g} second to settle; settled placement and objects leaving the workspace end tests early.</p>')
        for r in results:
            episode=folder/f'episode_{r["episode"]:03d}';comparison=episode/'comparison_external.mp4';cameras=episode/'cameras.mp4'
            video=comparison if comparison.exists() else cameras
            if video.exists():
                relative=escape(str(video.relative_to(a.run)));trace=escape(str((episode/'rollout.json').relative_to(a.run)))
                label='Synthetic expert / learned ACT, external view' if comparison.exists() else 'Learned ACT, wrist / external views'
                parts.append(f'<h3>Episode {r["episode"]}: {"success" if r["success"] else "failed"}</h3><p>{label} · <a href="{trace}">joint and contact trace</a></p><video controls preload="metadata" src="{relative}"></video>')
        parts.append('</section>')
    parts.append('<p>Simulation training and testing only. Geometry, contact and servo calibration affect hardware transfer. The tray has roughly 8 mm of clearance per side around an aligned cube.</p>')
    output=a.run/'index.html';output.write_text('\n'.join(parts)+'\n');print(output)


if __name__=='__main__':main()
