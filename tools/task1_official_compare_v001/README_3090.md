# Task1 temporary 3090 training workspace

SSH: `task1-3090`. Work directory:
`/mnt/file4/junyao/task1_official_compare_20261108`.
The Python 3.11 virtualenv is local to this directory. Package versions are in
`tools/task1_official_compare_v001/env-spec-3090.json`. PyTorch is 2.7.0+cu128,
torchvision 0.22.0+cu128, diffusers 0.31.0, robomimic 0.2.0 source.
GPUs 0, 2, and 3 belong to other running jobs and are excluded. A uses GPUs 4,5; B uses GPUs 6,7. GPU 1 remains available.

Both groups use the official hybrid architecture, FP32, global batch 64,
126700 updates to the 100-epoch comparison checkpoint, and a cosine schedule
over the unchanged 3801000-update full budget. This milestone is not the full
3000-epoch convergence result. A resumes its complete step-17332 checkpoint on two GPUs. B resumes its complete
step-6335 checkpoint on two GPUs, with local batch 32 and global batch 64.
B previously used four ranks. The retained ranks restore their saved RNG streams;
changing the partition changes the subsequent stochastic trajectory. Model, EMA,
AdamW, scheduler, update counter, and global data cursor are all restored. All 107 demonstrations and the 96/11
split remain unchanged. Each rank decodes its portion of a global batch from
the same memory-mapped compressed data. The final smaller batch uses sample-
count weighting, so DDP averaging preserves the global mean loss.

Preparation completes when `environment_installed.json` and `inputs_ready.json`
exist. No uncompressed RGB arrays are required by the following commands.

Run these commands sequentially from the work directory for the checkpoint-resume witness:

```bash
CUDA_VISIBLE_DEVICES=4,5 .venv/bin/python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 tools/task1_official_compare_v001/train_ddp.py --config configs/official.json --resume-check
CUDA_VISIBLE_DEVICES=6,7 .venv/bin/python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 tools/task1_official_compare_v001/train_ddp.py --config configs/current.json --resume-check
```

Each must exit zero and print `TASK1_DDP_RESUME_OK`. The corresponding
`outputs/<variant>/resume_probe/resume_result.json` must report two new updates,
global batch 64, world size 2, finite reloaded DDPM100 output `[1,8,7]`.
Expected start/end steps: A 17332/17334, B 6335/6337. Probe outputs are separate
from the formal checkpoints and do not advance formal training.
These are kernel/checkpoint witnesses, not task-success evaluations.

After the witnesses pass, launch:

```bash
.venv/bin/python -u tools/task1_official_compare_v001/run_3090.py
```

The supervisor starts both groups on disjoint GPUs, writes `training_status.json`,
and records GPU telemetry. Each trainer records 200 warmup updates plus 500
stable updates. Validation and complete checkpoint saves are timed separately.
Complete checkpoints store raw/EMA weights, optimizer, scheduler, epoch/batch
cursor, and per-rank RNG. Physical GPU IDs may change. A changed world size is recorded explicitly in
`training_start.json`; the global batch and data order are preserved. Stop signals
set a local request flag; all ranks agree on exit through the collective before saving.

Return and cleanup are performed from the original `isaac-b` repository by
`lifecycle_3090.py --monitor-only` over the server-to-server SSH connection after
the supervisor has been started. It copies every file in the completed return manifest,
loads both raw/EMA models and optimizer states, runs actual CUDA inference,
and only then deletes this exact temporary remote directory and revokes the
temporary transfer credential.
The original 107-trajectory dataset and step-497 checkpoint remain on `isaac-b`.
The returned checkpoints then receive 20 paired, teacher-free Task1 rollouts
per group in the original Isaac environment. No Isaac installation is needed here.
