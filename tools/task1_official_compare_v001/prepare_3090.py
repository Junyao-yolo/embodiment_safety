"""Relocate the two matched configurations into the temporary GPU workspace."""
import json
from pathlib import Path
root=Path(__file__).resolve().parents[2]
source=root/'outputs/runs/task1/diffusion_policy/official_input_compare_v001'
(root/'configs').mkdir(exist_ok=True)
for variant,gpus in [('official',[4,5]),('current',[6,7])]:
    cfg=json.loads((source/(variant+'.json')).read_text())
    cfg.update(data=str(root/'outputs/datasets/task1/diffusion_parallel_diverse_new_wrist_v001'),
        proprio_path=str(source/'proprio9.npy'),output_dir=str(root/'outputs'/variant),gpu_ids=gpus,
        distributed_world_size=len(gpus),initial_checkpoint=str(source/'official/checkpoints/last.pt') if variant=='official' else None,
        checkpoint_every=6335)
    (root/'configs'/(variant+'.json')).write_text(json.dumps(cfg,indent=2))
print('TASK1_CONFIGS_READY')
