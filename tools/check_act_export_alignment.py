#!/usr/bin/env python3
"""Check decoded training frames against source PNGs across video/shard boundaries."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',type=Path,default=Path('output/act_data/train'))
    p.add_argument('--source',type=Path,default=Path('output/pick_and_place'));p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    metadata=json.loads((a.dataset/'chrono_metadata.json').read_text())
    dataset=LeRobotDataset('local/so101-raised-train',root=a.dataset,video_backend='pyav')
    episodes=sorted(set([0,64,65,128,129,192,193,256,257,320,321,len(metadata['source_episodes'])-1]));report=[]
    for index in episodes:
        entry=metadata['source_episodes'][index]
        for frame in [0,123,299]:
            row=dataset[index*300+frame]
            for camera,alias in metadata['so101_conversion']['camera_names'].items():
                reference=np.asarray(Image.open(a.source/entry['source_episode_path']/'images'/camera/f'{frame:06d}.png')).astype(float)/255.
                actual=row['observation.images.'+alias].permute(1,2,0).numpy()
                error=float(abs(reference-actual).mean()*255)
                report.append(dict(episode=index,frame=frame,camera=camera,mean_absolute_rgb_difference_255=error))
                if error>5:raise ValueError(f'Decoded frame differs substantially from source: {report[-1]}')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(f'{len(report)} decoded frames checked across shard/video boundaries; maximum mean RGB difference {max(r["mean_absolute_rgb_difference_255"] for r in report):.3f}/255.')


if __name__=='__main__':main()
