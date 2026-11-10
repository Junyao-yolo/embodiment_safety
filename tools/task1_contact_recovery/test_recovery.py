"""Check takeover boundaries, release gates, and teacher segment provenance."""

import json
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from tasks.task1_dimm_insert.env.contact_recovery import ContactRecovery
from tools.task1_contact_recovery.prepare import pack


def recovery(tmp_path):
    spec=tmp_path/'spec.json'
    spec.write_text(json.dumps(dict(handoff_index=20,seed=1,teacher_max_seconds=120)))
    panel=SimpleNamespace(contact_metrics={})
    return ContactRecovery(panel,spec,tmp_path)


def test_student_retains_control_before_handoff(tmp_path):
    control=recovery(tmp_path)
    assert control.step(19,{},None,{}) is None
    assert control.teacher is None and control.error is None


def test_unstable_handoff_is_not_training_data(tmp_path):
    control=recovery(tmp_path)
    assert control.step(20,dict(sim_time=10),dict(evidence={}),{}) is None
    result=json.loads((tmp_path/'teacher_result.json').read_text())
    assert result['error']=='handoff_grasp_not_stable'
    assert not result['training_eligible']


def test_release_requires_actual_seating_and_closure(tmp_path):
    control=recovery(tmp_path)
    control.started=0;control.wall_start=time.monotonic()
    control.teacher=SimpleNamespace(state='RELEASE',error=None,action=lambda cfg:np.array([0,0,0,0,0,0,.1]))
    raw=dict(sim_time=1,task_truth=dict(poses=dict(dimm=[0,0,0,0,0,0,1])))
    action=control.step(21,raw,dict(evidence=dict(no_safety_violation=True),end=dict(seated_operational=False,closure_history=False)),{})
    assert action is None and control.error=='release_without_verified_seating'
    assert not json.loads((tmp_path/'teacher_result.json').read_text())['training_eligible']


def test_teacher_force_limit_excludes_training(tmp_path):
    control=recovery(tmp_path);control.started=0;control.wall_start=time.monotonic()
    control.teacher=SimpleNamespace(state='INSERT')
    control.panel.contact_metrics={'dimm_environment_N':6.}
    assert control.step(21,dict(sim_time=1),dict(evidence=dict(no_safety_violation=False)),{}) is None
    result=json.loads((tmp_path/'teacher_result.json').read_text())
    assert result['error']=='teacher_force_limit:dimm_environment_N'
    assert not result['training_eligible']


def test_pack_holds_out_all_variants_of_same_source_seed(tmp_path):
    cases=[]
    for k,seed in enumerate((11,11,22)):
        episode=tmp_path/f'episode{k}';episode.mkdir()
        (episode/'teacher_result.json').write_text(json.dumps(dict(training_eligible=True)))
        # Student prefix must not become a teacher target.
        (episode/'action_000.json').write_text(json.dumps(dict(action_source='student')))
        for i in range(1,18):
            (episode/f'action_{i:03d}.json').write_text(json.dumps(dict(action_source='teacher',executed=[0]*7,teacher_stage='ALIGN')))
            (episode/f'raw_{i:03d}.json').write_text(json.dumps(dict(ee_position_m=[0]*3,ee_quaternion_xyzw=[0,0,0,1],joint_positions=[0]*9)))
            state=np.zeros((1,31),np.float32);state[0,25]=1
            np.savez(episode/f'obs_{i:03d}.npz',state=state,**{key:np.full((1,256,256,3),i,np.uint8) for key in ('external','wrist_left','wrist_right')})
        cases.append(dict(episode=str(episode),seed=seed))
    out=tmp_path/'packed';manifest=pack(cases,out)
    assert manifest['rows']==51
    train={e['seed'] for e in manifest['episodes'] if e['split']=='train'}
    valid={e['seed'] for e in manifest['episodes'] if e['split']=='validation'}
    assert train=={11} and valid=={22}
    assert len(manifest['train'])==4 and len(manifest['validation'])==2
    rows=[json.loads(line) for line in (out/'rows.jsonl').read_text().splitlines()]
    assert all(row['source_index']>=1 for row in rows)
    assert np.load(out/'state.npy').shape==(51,31)
