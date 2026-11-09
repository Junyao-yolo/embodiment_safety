"""Train any configured task with the official Diffusion Policy Workspace."""

from learning.bootstrap import configure

configure()

import hydra
from omegaconf import OmegaConf
import torch

OmegaConf.register_new_resolver("eval", eval, replace=True)


@hydra.main(
    version_base=None,
    config_path="learning/config",
    config_name="train_diffusion_unet_hybrid_workspace",
)
def main(cfg):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = True
    workspace = hydra.utils.get_class(cfg._target_)(cfg)
    workspace.run()
    if workspace._saving_thread is not None:
        workspace._saving_thread.join()
    workspace.save_checkpoint(use_thread=False)


if __name__ == "__main__":
    main()
