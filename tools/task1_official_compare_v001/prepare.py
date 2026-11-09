"""Extract measured proprioception and write matched experiment configurations."""
import common
import argparse
import copy
import json
import math
from pathlib import Path
import time
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--wrist', default='wrist_right', choices=['wrist_left','wrist_right'])
    args = ap.parse_args()
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    data = common.ROOT/'outputs/datasets/task1/diffusion_parallel_diverse_new_wrist_v001'
    manifest = json.loads((data/'manifest.json').read_text())
    proprio = out/'proprio9.npy'
    if not proprio.exists():
        start = time.monotonic()
        rows = [json.loads(line) for line in (data/'rows.jsonl').open()]
        result = np.empty((manifest['rows'],9), np.float32)
        raw_state = np.load(data/'state.npy', mmap_mode='r')
        for episode in manifest['episodes']:
            offset = episode['offset']
            source = Path(episode['episode'])/'transitions.jsonl'
            wanted = {rows[i]['source_index']: i for i in range(offset,offset+episode['rows'])}
            found = 0
            with source.open() as stream:
                for line in stream:
                    item = json.loads(line)
                    index = item.get('index')
                    if index not in wanted: continue
                    target = wanted[index]
                    obs = item['observations']
                    np.testing.assert_allclose(obs['joint_positions'],raw_state[target,:9],rtol=1e-5,atol=1e-6)
                    result[target] = common.lowdim(obs); found += 1
            assert found==episode['rows'], (source, found, episode['rows'])
            print(json.dumps(dict(episode=episode['episode'], rows_done=offset+found)), flush=True)
        assert np.isfinite(result).all()
        np.save(proprio, result)
        common.write_json(out/'proprio_manifest.json',dict(rows=len(result),source=str(data),
            fields=['ee_position_m[3]','ee_quaternion_xyzw[4]','joint_positions[-2:]'],
            measured=True,extraction_seconds=time.monotonic()-start))
    template = json.loads((common.ROOT/'outputs/runs/robomimic/tool_hang_official_success_v001/config.json').read_text())
    steps_per_epoch = math.ceil(len(manifest['train'])/64)
    configs = []
    for variant in ('official','current'):
        cfg = dict(seed=42, input_variant=variant, data=str(data), proprio_path=str(proprio),
            output_dir=str(out/variant), batch_size=64, num_workers=8,
            epochs=3000, updates=steps_per_epoch*3000, scheduler_total_steps=steps_per_epoch*3000,
            stage_epochs=100, stage_updates=steps_per_epoch*100, steps_per_epoch=steps_per_epoch,
            lr=1e-4, warmup_steps=500, precision='float32', validate_every=steps_per_epoch*5,
            checkpoint_every=steps_per_epoch, cameras=['external',args.wrist] if variant=='official' else list(common.CAMERAS),
            action_interface='normalized_world_velocity_100ms', action_label='expert_clean',
            action_normalization='train_rows_limits_range_eps_1e-8', crop=[216,216],
            model_kind='official_hybrid', official_revision='5ba07ac6661db573af695b419a7947ecb704690f',
            optimizer=dict(name='AdamW',betas=[.95,.999],weight_decay=1e-6,eps=1e-8),
            ema=dict(power=.75), gradient_clipping=None,
            train_episodes=96, validation_episodes=11, windows=len(manifest['train']),
            image_normalization='minus1_plus1' if variant=='official' else 'zero_one',
            sequence_sampling='existing_unpadded_Task1_windows', control_hz=10,
            input_note='Task1 camera content and world frame; official observation form, not ToolHang observations')
        policy = copy.deepcopy(template['official_policy'])
        size = 240 if variant=='official' else 256
        obs = {key:dict(shape=[3,size,size],type='rgb') for key in cfg['cameras']}
        if variant=='official':
            obs.update(robot0_eef_pos=dict(shape=[3]),robot0_eef_quat=dict(shape=[4]),robot0_gripper_qpos=dict(shape=[2]))
        else: obs['state'] = dict(shape=[33])
        policy['shape_meta'] = dict(action=dict(shape=[7]),obs=obs)
        cfg['official_policy'] = policy
        path = out/f'{variant}.json'; common.write_json(path,cfg); configs.append(str(path))
    common.write_json(out/'plan.json', dict(configs=configs,stage_epochs=100,total_epochs=3000,
        sequence=['official_train','current_train','official_eval20','current_eval20'],
        auto_continue_after_comparison=False,eval_seeds=list(range(2026110301,2026110321)),
        evaluation=dict(max_simulation_steps=3000,max_wall_seconds=2700,action_steps=8,
            teacher_enabled=False,safety_mode='warn',checkpoint_selection='fixed_100_epoch_ema'),
        data=dict(episodes=107,train=96,validation=11,train_windows=len(manifest['train']),
                  validation_windows=len(manifest['validation'])),
        contrasts=['camera_count','image_resolution','lowdim_state','input_normalization'],
        note='Official architecture/hyperparameters adapted to Task1; first comparison is a 100-epoch milestone, not 3000-epoch convergence.'))
    print(json.dumps(dict(configs=configs,steps_per_epoch=steps_per_epoch)),flush=True)


if __name__=='__main__': main()
