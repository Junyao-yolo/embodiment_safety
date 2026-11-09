"""ToolHang benchmark using the repository's Diffusion Policy implementation."""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE / 'deps'), str(ROOT / 'algos/diffusion_policy/vendor')]
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '4'
os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
os.environ.setdefault('WANDB_MODE', 'disabled')

CAMERAS = ('sideview_image', 'robot0_eye_in_hand_image')
SHAPE = dict(action=dict(shape=[10]), obs={
    **{k: dict(shape=[3, 240, 240], type='rgb') for k in CAMERAS},
    'robot0_eef_pos': dict(shape=[3]),
    'robot0_eef_quat': dict(shape=[4]),
    'robot0_gripper_qpos': dict(shape=[2]),
})
DEFAULT = dict(seed=42, updates=100000, batch_size=16, num_workers=4,
               horizon=16, n_obs_steps=2, n_action_steps=8, num_inference_steps=16,
               down_dims=[128, 256, 512], rgb_features=128, diffusion_step_embed_dim=128,
               lr=1e-4, warmup_steps=500, prediction_type='sample',
               val_ratio=0.02, validate_every=5000, checkpoint_every=5000,
               eval_every=25000, eval_episodes=20, max_episode_steps=700,
               precision='bfloat16', abs_action=True,
               dataset_path=str(ROOT / 'outputs/datasets/robomimic/tool_hang/ph/image_abs.hdf5'),
               output_dir=str(ROOT / 'outputs/runs/robomimic/tool_hang_dp_bc_v001'))


def build(cfg):
    if cfg.get('model_kind') == 'official_hybrid':
        sys.path.insert(0, str(HERE / 'official_deps'))
        import hydra
        from omegaconf import OmegaConf
        return hydra.utils.instantiate(OmegaConf.create(cfg['official_policy']))
    import torch
    from torchvision.models import resnet18
    from diffusers import DDIMScheduler
    from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
    from diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy
    rgb = resnet18(weights=None)
    rgb.fc = torch.nn.Linear(rgb.fc.in_features, cfg['rgb_features'])
    encoder = MultiImageObsEncoder(SHAPE, rgb, use_group_norm=True,
                                  share_rgb_model=False, imagenet_norm=False)
    scheduler = DDIMScheduler(num_train_timesteps=100, beta_schedule='squaredcos_cap_v2',
                             clip_sample=True, prediction_type=cfg['prediction_type'])
    return DiffusionUnetImagePolicy(SHAPE, scheduler, encoder,
        **{k:cfg[k] for k in ('horizon','n_obs_steps','n_action_steps','num_inference_steps',
                             'diffusion_step_embed_dim','down_dims')})


def cuda_batch(batch):
    from diffusion_policy.common.pytorch_util import dict_apply
    return dict_apply(batch, lambda x: x.cuda(non_blocking=True))


def dataset(cfg):
    from diffusion_policy.dataset.robomimic_replay_image_dataset import RobomimicReplayImageDataset
    return RobomimicReplayImageDataset(shape_meta=SHAPE, dataset_path=cfg['dataset_path'],
        horizon=cfg['horizon'], pad_before=cfg['n_obs_steps']-1, pad_after=cfg['n_action_steps']-1,
        n_obs_steps=cfg['n_obs_steps'], abs_action=True, use_cache=True,
        val_ratio=cfg['val_ratio'], seed=cfg['seed'])
