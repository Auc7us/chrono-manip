#!/usr/bin/env python3
"""Copy a standard ACT inference bundle with the selected execution settings."""
import argparse
import json
from pathlib import Path
import shutil


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('checkpoint',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--dataset',type=Path,default=Path('output/act_data/train'))
    p.add_argument('--temporal-ensemble-coeff',type=float,default=.01);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    shutil.copytree(a.checkpoint,a.output,ignore=shutil.ignore_patterns('train_config.json'))
    config_path=a.output/'config.json';config=json.loads(config_path.read_text())
    config['temporal_ensemble_coeff']=a.temporal_ensemble_coeff;config['n_action_steps']=1
    config_path.write_text(json.dumps(config,indent=2)+'\n')
    metadata=json.loads((a.dataset/'chrono_metadata.json').read_text())
    calibration={key:metadata[key] for key in ['so101_conversion','action_semantics','camera_config','tray_geometry','robot_type']}
    (a.output/'chrono_calibration.json').write_text(json.dumps(calibration,indent=2)+'\n')
    provenance=dict(source_training_checkpoint=str(a.checkpoint.resolve()),weight_changes=False,
        execution=dict(n_action_steps=1,temporal_ensemble_coeff=a.temporal_ensemble_coeff),
        selection='Overlapping-chunk ensembling selected using recorded held-out layouts, before fresh-layout testing',
        resume_training_from='Use the original training checkpoint, which retains optimizer state')
    (a.output/'deployment.json').write_text(json.dumps(provenance,indent=2)+'\n');print(a.output)


if __name__=='__main__':main()
