#!/usr/bin/env python3
"""Export the ACT network (normalized state and images -> normalized action chunk) to ONNX for TensorRT.

LeRobot's saved pre/post-processors and the temporal ensembler stay in Python around the engine.
Also writes one random input set and the PyTorch output as raw float32 files, so `trtexec
--loadInputs` on the target can check the engine against PyTorch.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    import torch
    from act_inference import load_act
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('checkpoint',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--height',type=int,default=480);p.add_argument('--width',type=int,default=640);p.add_argument('--opset',type=int,default=17)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    policy,_,_=load_act(a.checkpoint,device='cpu');config=policy.config;keys=list(config.image_features)

    class Core(torch.nn.Module):
        def __init__(self):
            super().__init__();self.model=policy.model
        def forward(self,state,*images):
            return self.model({'observation.state':state,'observation.images':list(images)})[0]

    core=Core().eval();names=['state',*[k.split('.')[-1] for k in keys]]
    torch.manual_seed(0)
    inputs=(torch.randn(1,config.robot_state_feature.shape[0]),*[torch.randn(1,3,a.height,a.width) for _ in keys])
    with torch.no_grad():reference=core(*inputs)
    a.output.mkdir(parents=True)
    torch.onnx.export(core,inputs,str(a.output/'act_core.onnx'),input_names=names,output_names=['actions'],opset_version=a.opset,dynamo=False)
    check=a.output/'check';check.mkdir()
    for name,tensor in zip(names,inputs):tensor.numpy().astype(np.float32).tofile(check/f'{name}.bin')
    reference.numpy().astype(np.float32).tofile(check/'actions_pytorch.bin')
    (a.output/'act_core.json').write_text(json.dumps(dict(checkpoint=str(a.checkpoint.resolve()),inputs=dict(zip(names,['observation.state',*keys])),
        input_shapes={n:list(t.shape) for n,t in zip(names,inputs)},output='actions',output_shape=list(reference.shape),
        semantics='inputs are after the saved preprocessor (normalized); output goes through the saved postprocessor',
        trtexec_check=' '.join(f'{n}:check/{n}.bin' for n in names)),indent=2)+'\n')
    print(f'Wrote {a.output}/act_core.onnx; inputs {names}, output {list(reference.shape)}')


if __name__=='__main__':main()
