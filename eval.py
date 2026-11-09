"""Evaluate a Workspace checkpoint through its configured EnvRunner."""

from learning.bootstrap import configure

configure()

import argparse
import json
from pathlib import Path
import dill
import hydra
from omegaconf import OmegaConf
import torch

OmegaConf.register_new_resolver("eval", eval, replace=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--checkpoint", type=Path, required=True)
    parser.add_argument("-o", "--output-dir", type=Path, required=True)
    parser.add_argument("-d", "--device", default="cuda:0")
    parser.add_argument("overrides", nargs="*", help="Hydra-style key=value overrides")
    args = parser.parse_args()
    if any("=" not in item for item in args.overrides):
        parser.error("Overrides must use key=value")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = True
    payload = torch.load(
        args.checkpoint, map_location="cpu", pickle_module=dill, weights_only=False
    )
    cfg = OmegaConf.merge(payload["cfg"], OmegaConf.from_dotlist(args.overrides))
    workspace = hydra.utils.get_class(cfg._target_)(
        cfg, output_dir=str(args.output_dir)
    )
    workspace.load_payload(payload, exclude_keys=("optimizer",), include_keys=())
    del payload
    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    policy.to(args.device).eval()
    runner = hydra.utils.instantiate(
        cfg.task.env_runner, output_dir=str(args.output_dir.resolve())
    )
    OmegaConf.save(cfg, args.output_dir / "eval_config.yaml")
    metrics = runner.run(policy)
    report = dict(
        checkpoint=str(args.checkpoint.resolve()),
        weights="ema" if cfg.training.use_ema else "model",
        metrics=metrics,
    )
    (args.output_dir / "eval_log.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
