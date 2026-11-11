"""Warm-start the official Workspace without restoring an obsolete optimizer."""
import torch
from diffusion_policy.workspace.train_diffusion_unet_hybrid_workspace import TrainDiffusionUnetHybridWorkspace

class FinetuneDiffusionUnetHybridWorkspace(TrainDiffusionUnetHybridWorkspace):
    def __init__(self,cfg,output_dir=None):
        super().__init__(cfg,output_dir)
        payload=torch.load(cfg.pretrained_checkpoint,map_location="cpu",weights_only=False)
        self.model.load_state_dict(payload["model"],strict=True)
        self.ema_model.load_state_dict(payload["model"],strict=True)
        self.epoch=1
