#!/usr/bin/env python3
"""Run `lerobot-record` with ACT's network replaced by a TensorRT engine from tools/export_act_onnx.py.

LeRobot still loads --policy.path for its config, pre/post-processors and temporal ensembler; only
ACTPolicy.predict_action_chunk runs on the engine. Usage (on the robot computer):
  ACT_TRT_ENGINE=~/trt_v1/act_core_fp16.engine python trt_act_record.py [lerobot-record args]
  python trt_act_record.py --check POLICY_DIR ENGINE   # engine vs PyTorch on random inputs; no robot
"""
import json
import os
from pathlib import Path
import sys
import time


class Engine:
    def __init__(self,path):
        import tensorrt as trt
        import torch
        self.torch=torch;self.path=Path(path).expanduser()
        self.meta=json.loads((self.path.parent/'act_core.json').read_text())
        runtime=trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self.engine=runtime.deserialize_cuda_engine(self.path.read_bytes())
        if self.engine is None:raise RuntimeError(f'Could not load TensorRT engine {self.path}')
        self.context=self.engine.create_execution_context()
        self.output=torch.empty(self.meta['output_shape'],dtype=torch.float32,device='cuda')
        self.context.set_tensor_address('actions',self.output.data_ptr());self.stream=torch.cuda.Stream()

    def __call__(self,batch):
        torch=self.torch;held=[]
        for name,key in self.meta['inputs'].items():
            tensor=batch[key].to('cuda',torch.float32).contiguous()
            if list(tensor.shape)!=self.meta['input_shapes'][name]:
                raise ValueError(f'{key}: engine expects {self.meta["input_shapes"][name]}, got {list(tensor.shape)}')
            held.append(tensor);self.context.set_tensor_address(name,tensor.data_ptr())
        # TensorRT warns on the default stream; run on our own after the input copies finish.
        self.stream.wait_stream(torch.cuda.current_stream())
        if not self.context.execute_async_v3(self.stream.cuda_stream):raise RuntimeError('TensorRT execution failed')
        self.stream.synchronize();return self.output.clone()


def patch(path):
    from lerobot.policies.act.modeling_act import ACTPolicy
    engine=Engine(path)

    def predict_action_chunk(self,batch,*args,**kwargs):
        keys=set(engine.meta['inputs'].values())-{'observation.state'}
        if keys!=set(self.config.image_features):raise ValueError(f'Engine cameras {sorted(keys)} != policy cameras {sorted(self.config.image_features)}')
        return engine(batch)

    ACTPolicy.predict_action_chunk=predict_action_chunk
    print(f'ACT inference on TensorRT engine {engine.path}',flush=True)
    return engine


def check(policy_dir,engine_path):
    import torch
    from lerobot.policies.act.modeling_act import ACTPolicy
    policy=ACTPolicy.from_pretrained(str(Path(policy_dir).expanduser())).to('cuda').eval();engine=Engine(engine_path)
    torch.manual_seed(1)
    batch={key:torch.randn(engine.meta['input_shapes'][name],device='cuda') for name,key in engine.meta['inputs'].items()}
    with torch.inference_mode():reference=ACTPolicy.predict_action_chunk(policy,batch)
    result=engine(batch);print(f'max |engine - pytorch| {float((result-reference).abs().max()):.4f} (normalized units)')
    for name,run in [('pytorch',lambda:ACTPolicy.predict_action_chunk(policy,batch)),('tensorrt',lambda:engine(batch))]:
        with torch.inference_mode():
            for _ in range(10):run()
            torch.cuda.synchronize();times=[]
            for _ in range(100):
                t=time.perf_counter();run();torch.cuda.synchronize();times.append(1000*(time.perf_counter()-t))
        times.sort();print(f'{name:9s} predict_action_chunk median {times[50]:.1f} ms, p90 {times[90]:.1f} ms')


if __name__=='__main__':
    if sys.argv[1:2]==['--check']:check(*sys.argv[2:4]);sys.exit()
    if 'ACT_TRT_ENGINE' not in os.environ:sys.exit('Set ACT_TRT_ENGINE to the .engine file')
    patch(os.environ['ACT_TRT_ENGINE'])
    from lerobot.scripts.lerobot_record import main
    main()
