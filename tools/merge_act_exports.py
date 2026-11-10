#!/usr/bin/env python3
"""Join independent official LeRobot exports and preserve Chrono provenance."""
import argparse
import json
from pathlib import Path


def main():
    from lerobot.datasets.aggregate import aggregate_datasets
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path)
    p.add_argument('shards',nargs='+',type=Path);a=p.parse_args()
    metadata=[json.loads((s/'chrono_metadata.json').read_text()) for s in a.shards]
    reference={k:v for k,v in metadata[0].items() if k!='source_episodes'}
    if any({k:v for k,v in m.items() if k!='source_episodes'}!=reference for m in metadata):
        raise ValueError('Shards use different calibration, cameras, units or physics')
    episodes=[e for m in metadata for e in m['source_episodes']]
    indices=[e['source_manifest_index'] for e in episodes]
    if len(indices)!=len(set(indices)):raise ValueError('Duplicate source episodes in export shards')
    aggregate_datasets([f'local/so101-raised-train-shard-{i}' for i in range(len(a.shards))],
        'local/so101-raised-train',roots=a.shards,aggr_root=a.output)
    info=json.loads((a.output/'meta/info.json').read_text())
    if info['total_episodes']!=len(episodes):raise ValueError('Merged episode count mismatch')
    reference['source_episodes']=episodes
    (a.output/'chrono_metadata.json').write_text(json.dumps(reference,indent=2)+'\n')
    print(f'Merged {len(episodes)} episodes / {info["total_frames"]} frames locally into {a.output}')


if __name__=='__main__':main()
