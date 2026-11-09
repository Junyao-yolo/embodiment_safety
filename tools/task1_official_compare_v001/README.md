# Task1 official DP-C input comparison

Working directory: `/mnt/data2/wjy_SERL/Embodied_Safety` on `isaac-b`.
Python: `algos/diffusion_policy/.venv/bin/python`. Dependencies are process-local in
`tools/tool_hang_benchmark_v001/official_deps` and `deps`; no base environment changes.
GPU: RTX 4090. Training and evaluation run sequentially and wait for other GPU compute jobs.

Both models use the official hybrid image policy at vendor revision
`5ba07ac6661db573af695b419a7947ecb704690f`: independent ResNet18 encoders,
GroupNorm, SpatialSoftmax, U-Net widths 512/1024/2048, epsilon prediction,
DDPM100, horizon 16, two observations, eight executed actions. FP32 batch 64,
AdamW 1e-4, betas 0.95/0.999, decay 1e-6, EMA power 0.75,
500-step warmup and cosine schedule over 3000 epochs. No gradient clipping.

The official-input group uses external and right wrist RGB, resized to 240 square,
training random crop / evaluation center crop 216, image normalization [-1,1],
measured world-frame EEF position, XYZW quaternion, and two finger positions (9D).
The current-input group uses all three 256-square images, the same 216 crop,
image normalization [0,1], and the existing rotation6d33 state, including causal
progress bits. This is a comparison of input packages, not a single-variable ablation.

The 107 accepted new-wrist trajectories retain their existing 96/11 whole-episode
split and 81049/9188 unpadded windows. Both groups predict the same 7D Task1
expert_clean world-velocity command, held for 100 ms. Training-only action
range normalization preserves small nonconstant ranges with range_eps=1e-8.
These action semantics and unpadded Task1 windows differ from ToolHang absolute
pose commands and padded sequences. This is an official-architecture Task1
adaptation, not an exact ToolHang reproduction.

Preparation:

```bash
algos/diffusion_policy/.venv/bin/python -u tools/task1_official_compare_v001/prepare.py --output outputs/runs/task1/diffusion_policy/official_input_compare_v001
```

GPU kernel and checkpoint witnesses (run sequentially):

```bash
algos/diffusion_policy/.venv/bin/python -u tools/task1_official_compare_v001/smoke.py --config outputs/runs/task1/diffusion_policy/official_input_compare_v001/official.json
algos/diffusion_policy/.venv/bin/python -u tools/task1_official_compare_v001/smoke.py --config outputs/runs/task1/diffusion_policy/official_input_compare_v001/current.json
```

Each command must print `TASK1_OFFICIAL_SMOKE_OK`, execute two batch64 FP32
gradient updates, reload EMA, and produce a finite [1,8,7] DDPM action tensor.
Results and smoke checkpoints are stored under each variant's `smoke/`.

Live IPC/control witness (8 steps, not a success-rate test):

```bash
algos/diffusion_policy/.venv/bin/python -u tools/task1_official_compare_v001/evaluate.py --checkpoint outputs/runs/task1/diffusion_policy/official_input_compare_v001/official/smoke/checkpoint.pt --output outputs/runs/task1/diffusion_policy/official_input_compare_v001/official/smoke/live8 --episodes 1 --max-steps 8 --wall-seconds 600
```

Run both 100-epoch training milestones and then 20 matched episodes per group:

```bash
algos/diffusion_policy/.venv/bin/python -u tools/task1_official_compare_v001/pipeline.py --run outputs/runs/task1/diffusion_policy/official_input_compare_v001
```

Each first-stage training budget is 126700 updates; the full 3000-epoch budget
and scheduler are 3801000 updates per group. First-stage results are not full
convergence. Complete checkpoints include raw/EMA weights, optimizer, scheduler,
RNG, epoch and batch cursor. Re-running a training command resumes from `last.pt`.
SIGTERM/SIGINT on the trainer requests a full checkpoint at the next boundary.

Full evaluation uses paired diffusion seeds 2026110301–2026110320 and the
unchanged default scene reset, 3000 control steps (300 simulated seconds),
2700-second wall limit, no teacher, and fixed 100-epoch EMA checkpoints.
Both completion under warning-only safety mode and strict full-task success
are reported separately, with stage rates, simulation/wall/inference times.
Videos use recorded 10 Hz observations at 10 fps and represent simulation time.
GPU telemetry is in `gpu.jsonl`; `timing.json` measures updates 201–700 after
200 warmup updates. Validation and checkpoint saving are timed separately.

## Interpretation

The camera content, world coordinate frame, and controller remain Task1-specific.
The two input groups differ in camera count, resolution, low-dimensional state,
and normalization. The 20 rollouts measure paired policy randomness on the
default scene; they do not establish generalization to randomized geometry.
