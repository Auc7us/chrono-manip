#!/usr/bin/env python3
"""Compare held-out expert recordings with learned ACT external-camera rollouts."""
import argparse
import json
from pathlib import Path
import subprocess
import cv2
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('rollouts',type=Path)
    p.add_argument('--source',type=Path,default=Path('output/pick_and_place'));a=p.parse_args()
    results=json.loads((a.rollouts/'results.json').read_text())
    for result in results:
        folder=a.rollouts/f'episode_{result["episode"]:03d}';policy=folder/'cameras.mp4'
        if not result['source_layout'] or not policy.exists():continue
        expert=cv2.VideoCapture(str(a.source/result['source_layout']/'right_45.mp4'))
        learned=cv2.VideoCapture(str(policy));fps=result['fps']
        if not expert.isOpened() or not learned.isOpened():raise RuntimeError('Missing comparison input video')
        width=int(expert.get(cv2.CAP_PROP_FRAME_WIDTH));height=int(expert.get(cv2.CAP_PROP_FRAME_HEIGHT))
        raw=folder/'comparison_external_raw.mp4';writer=cv2.VideoWriter(str(raw),cv2.VideoWriter_fourcc(*'mp4v'),fps,(2*width,height+32))
        if not writer.isOpened():raise RuntimeError('Comparison video writer did not open')
        last=None;index=0
        try:
            while True:
                ok,frame=learned.read()
                if not ok:break
                ok,reference=expert.read()
                if ok:last=reference
                if last is None:raise RuntimeError('Expert reference video is empty')
                canvas=np.zeros((height+32,width*2,3),np.uint8);canvas[32:,:width]=last;canvas[32:,width:]=frame[32:height+32,width:2*width]
                reference_label='Synthetic expert' if ok else 'Synthetic expert (last frame)'
                for x,label in [(12,reference_label),(width+12,'Learned ACT')]:
                    cv2.putText(canvas,f'{label} | {index/fps:.2f} s',(x,22),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1)
                writer.write(canvas);index+=1
        finally:writer.release();expert.release();learned.release()
        subprocess.run(['ffmpeg','-y','-v','error','-i',str(raw),'-c:v','libx264','-threads','2','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(folder/'comparison_external.mp4')],check=True)
        print(folder/'comparison_external.mp4',flush=True)


if __name__=='__main__':main()
