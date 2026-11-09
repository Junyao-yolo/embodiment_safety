"""Import existing official image-policy weights into a configured Workspace."""

from learning.bootstrap import configure

configure()

import argparse
from pathlib import Path
import dill
import hydra
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import torch

OmegaConf.register_new_resolver("eval", eval, replace=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with initialize_config_dir(
        version_base=None, config_dir=str(Path(__file__).parent / "config")
    ):
        cfg = compose(
            "train_diffusion_unet_hybrid_workspace", overrides=[f"task={args.task}"]
        )
    source = torch.load(args.source, map_location="cpu", weights_only=False)
    state = source["ema"] if "ema" in source else source["model"]
    policy = hydra.utils.instantiate(cfg.policy)
    policy.load_state_dict(state, strict=True)
    state = policy.state_dict()
    step = int(source.get("step", 0))
    payload = dict(
        cfg=cfg,
        state_dicts={"model": state, "ema_model": state},
        pickles={"global_step": dill.dumps(step), "epoch": dill.dumps(0)},
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output, pickle_module=dill)
    print(f"Inference checkpoint: {args.output}")
    print("Imported weights only; optimizer and training schedule are not resumed.")


if __name__ == "__main__":
    main()
