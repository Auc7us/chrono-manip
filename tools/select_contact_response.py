#!/usr/bin/env python3
"""Choose effective compliance by loaded-jaw tracking on training recordings."""
import argparse
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--candidates',type=Path,nargs='+',required=True)
    p.add_argument('--episodes',type=int,nargs='+',default=[0,2,7]);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if any(ep>=8 for ep in a.episodes):raise ValueError('Episodes 8 and 9 are held out')
    with np.load(ROOT/'output/video_fit/reference_trajectory.npz') as data:states=data['observation_state'];actions=data['action']
    candidates=[];reference=None
    for directory in a.candidates:
        errors=[];rows=[];compliance=None
        for ep in a.episodes:
            source=directory/f'episode_{ep:03d}/physics_replay.json';physical=json.loads(source.read_text());mapping=physical['joint_map']
            signature=np.r_[mapping['scale'],mapping['offset_deg'],np.array(physical['base_world_from_cad']).ravel(),np.array(physical['wrist_profile']['link_from_optical']).ravel()]
            if reference is None:reference=signature
            else:np.testing.assert_allclose(signature,reference,atol=1e-10,rtol=0)
            compliance=physical.get('contact_compliance_m_n',0.);episode_errors=[]
            for row in physical['records']:
                i=row['dataset_index'];state=states[i,5]
                if actions[i,5]<1 and 5<state<20:
                    episode_errors.append(row['jaw_actual_deg']-(state*mapping['scale'][5]+mapping['offset_deg'][5]))
            if not episode_errors:raise ValueError(f'No loaded samples in {source}')
            errors.extend(episode_errors)
            rows.append(dict(episode=ep,samples=len(episode_errors),loaded_jaw_rms_deg=float(np.sqrt(np.mean(np.square(episode_errors)))),
                maximum_contact_force_n=physical['maximum_contact_force_n'],maximum_penetration_m=physical['maximum_penetration_m'],source=str(source)))
        candidates.append(dict(normal_compliance_m_per_n=compliance,loaded_jaw_rms_deg=float(np.sqrt(np.mean(np.square(errors)))),samples=len(errors),episodes=rows))
    selected=min(candidates,key=lambda row:row['loaded_jaw_rms_deg'])
    report=dict(method='effective contact-compliance candidate selected by recorded loaded-jaw tracking; no endpoint objective',
        training_episodes=a.episodes,held_out_episodes=[8,9],normal_compliance_m_per_n=selected['normal_compliance_m_per_n'],
        tangential_compliance_ratio=.25,damping_time_s=.002,training_loaded_jaw_rms_deg=selected['loaded_jaw_rms_deg'],
        candidates=candidates,material_parameters_independently_measured=False,
        assumptions=['Effective contact compliance represents omitted printed-jaw and arm compliance; not measured PLA stiffness',
            'Existing friction, jaw torque saturation and CAD collision geometry are retained',
            'Later images and final cube positions never drive simulated cube motion',
            'Large contact-force/penetration outliers remain visible and are not used as a calibrated force response'])
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['normal_compliance_m_per_n','training_loaded_jaw_rms_deg','training_episodes']},indent=2))


if __name__=='__main__':main()
