# 统一 Diffusion Policy 框架

项目：`/mnt/data2/wjy_SERL/Embodied_Safety`。所有命令在 ufdwaa 的项目根目录执行。

Task1、Task2 共用官方模型和训练循环；任务配置决定观测形状、动作含义、数据集和环境执行器。后续任务沿用同一接口。

## 与官方框架的对应

| 官方接口 | 本项目 |
| --- | --- |
| Hydra train.py → Workspace.run() | 项目根目录 train.py |
| eval.py → checkpoint cfg → EnvRunner.run(policy) | 项目根目录 eval.py |
| TrainDiffusionUnetHybridWorkspace | 直接使用 vendor 官方类 |
| DiffusionUnetHybridImagePolicy | 直接使用 vendor 官方类 |
| BaseImageDataset | learning/dataset/task1_image_dataset.py、task2_image_dataset.py |
| BaseImageRunner | learning/env_runner/task1_image_runner.py、task2_image_runner.py |
| task YAML | learning/config/task/*.yaml |
| shape_meta | 各 task YAML 定义图像、状态与动作维度 |
| 标准 Workspace checkpoint | cfg、state_dicts、pickles；训练保存 model、EMA 与优化器状态 |

官方源码版本：`5ba07ac6661db573af695b419a7947ecb704690f`。Policy、Workspace、ReplayBuffer、SequenceSampler、Normalizer 均从 `third_party/diffusion_policy` 导入。任务内保留真实物理执行、验收与安全测量。动作如何交给机器人由 `learning/controller` 负责。

## 已接入任务

| 配置 | 观测 | 动作 | 数据划分 |
| --- | --- | --- | --- |
| task1_image | 3 路 RGB256；state33 | 7 维末端/夹爪速度命令 | 107 条，96 train / 11 validation |
| task1_official_image | 外部与右腕 RGB240；EEF 位置3、四元数4、手指位置2 | 同一 7 维动作 | 同一 96 / 11 |
| task2_image | 3 路 RGB128；state23 | 48 维，每个 100 ms 包含 6 条关节/夹爪指令 | 100 条，91 train / 9 validation |

所有配置使用 2 帧观测。Task1 原始 state31 的目标四元数编码为 rotation6d，得到 state33；4 个进度标记保留因果时间对齐。Task1 当前输入图像归一化为 [0,1]，official 输入图像归一化为 [-1,1]。official 表示采用其观测组织形式，场景与动作仍属于 Task1。

Task1 动作保持 100 ms，7 个通道是归一化的世界坐标系线速度、角速度和夹爪速度。训练集提供 81,049 个窗口，验证集 9,188 个窗口，保持原有不填充采样。

Task2 原始示范动作是绝对关节目标；训练标签将 7 个关节转换为逐指令增量，夹爪仍为绝对宽度。执行器从上一实际执行目标累加重建 60 Hz 命令。48 维动作是一个 100 ms 控制块，`n_action_steps=4` 表示执行 4 个这样的块，即 0.4 秒后再预测。默认 8 块为 0.8 秒。物理频率为 120 Hz。默认 horizon16、前填充1、后填充7，共 163,359 / 14,468 个训练/验证窗口；改变动作块长度会改变后填充数量。

Task1 按完整回合划分；Task2 按共享初态组划分，防止同一初态的不同路径同时进入训练和验证。归一化仅使用训练回合。动作维数由任务决定，不要求不同任务具有相同维数。

## 训练

```bash
cd /mnt/data2/wjy_SERL/Embodied_Safety

algos/diffusion_policy/.venv/bin/python -B train.py \
  task=task1_image training.num_epochs=100

algos/diffusion_policy/.venv/bin/python -B train.py \
  task=task2_image training.num_epochs=100
```

Task1 的另一输入配置使用 `task=task1_official_image`。指定输出目录或动作块：

```bash
algos/diffusion_policy/.venv/bin/python -B train.py \
  task=task2_image training.num_epochs=100 n_action_steps=4 \
  hydra.run.dir=/mnt/data2/wjy_SERL/Embodied_Safety/outputs/runs/task2_riser_io/diffusion_policy/run_001
```

每次实验使用新的输出目录。查看合成配置而不启动训练：

```bash
algos/diffusion_policy/.venv/bin/python -B train.py task=task2_image --cfg job
```

共用配置：`learning/config/train_diffusion_unet_hybrid_workspace.yaml`。

| 参数 | 默认值 |
| --- | --- |
| 模型 | Hybrid image policy，ResNet 图像编码器 + conditional U-Net |
| U-Net down_dims | [512, 1024, 2048] |
| horizon / n_obs_steps / n_action_steps | 16 / 2 / 8 |
| 扩散训练/推理步数 | DDPM 100 / 100，epsilon target |
| batch / precision | 64 / FP32 |
| 优化器 | AdamW，lr=1e-4，betas=[0.95,0.999]，weight_decay=1e-6 |
| 学习率 | cosine，warmup 500 updates |
| EMA | power=0.75，max_value=0.9999 |
| epochs | 100 |
| rollout / checkpoint | 每 50 个 epoch，官方零起始编号，包括 epoch0 |
| validation / action sample | 每 1 / 5 个 epoch |
| 默认 rollout 数 | 每次 3 回合 |

`training.num_epochs` 同时影响本次训练预算和 cosine 调度长度。训练结束保存 `checkpoints/latest.ckpt`；top-k 按 rollout 的 `test/mean_score` 保存。该指标是任务完成率，不是训练 loss。查看 `logs.json.txt` 中的 train_loss、val_loss 与 action MSE；低离线误差不代表闭环成功。

## 评估

```bash
algos/diffusion_policy/.venv/bin/python -B eval.py \
  -c outputs/checkpoints/task2_riser_io/official_dp_epoch018.ckpt \
  -o outputs/runs/task2_riser_io/diffusion_policy/eval_001 \
  task.env_runner.n_test=20 task.env_runner.test_start_seed=990000 \
  n_action_steps=4
```

任务与模型从检查点配置恢复，默认加载 EMA。覆盖参数适合调整回合数量、执行预算、动作块、测试种子；更改模型形状需要相应权重。

Task1 使用 `task.env_runner.max_steps`，每步100 ms；Task2 使用 `task.env_runner.max_seconds`。Task1 每回合以独立 Isaac 进程复位；Task2 在同一进程内复用场景和相机。执行的是策略预测，任务真值仅用于环境测量与验收。

训练时的 rollout 用于模型选择；正式报告使用独立输出目录与未参与选模的测试种子。Task1 当前默认场景固定，种子改变策略采样，不能将其称为随机几何泛化。Task2 的种子改变物体初态；测试种子每次递增4以切换初态组。

输出：

- `eval_config.yaml`：本次模型与执行配置。
- `eval_log.json`：检查点、权重选择及指标。
- `rollouts/round_NNN/summary.json`：任务完成、安全完成与逐回合统计。
- Task1：`trial_NNN/eval/episode_01` 保存动作和观测；`trial_NNN/replay/actual_policy_replay.mp4` 保存视频。
- Task2：`trial_NNN/result.json`、`measurements.npz`、`rollout.mp4` 保存验收、测量及视频。

`test/mean_score` 表示任务成功率；`test/safe_completion_rate` 表示任务完成且满足已有安全判据的比例。Task2 另记录 `test/boundary_violation_rate`。任务判定保持各自物理含义；程序/传感器错误与策略未完成任务分开记录。

## 数据准备

当前 Task1 数据：

`/mnt/data2/wjy_SERL/Embodied_Safety/outputs/datasets/task1/diffusion_parallel_diverse_new_wrist_v001`

当前 Task2 数据：

`/mnt/data2/wjy_SERL/Embodied_Safety/outputs/task2_riser_io/data/opposed_v002/prepared`

追加数据后，准备新的目录，保留现有实验划分：

```bash
algos/diffusion_policy/.venv/bin/python -B -m tasks.task1_dimm_insert.data.prepare_diffusion \
  --source /绝对路径/Task1采集目录 --output /绝对路径/新Task1数据目录 \
  --expected 107 --label expert_clean

algos/diffusion_policy/.venv/bin/python -B -m tasks.task2_riser_io.data.prepare_diffusion \
  --source /绝对路径/Task2原始回合目录 --output /绝对路径/新Task2数据目录
```

Task1 准备器生成 state、action、proprio9、RGB、回合清单和无损 RGB 缓存。Task2 准备器生成其 Dataset 使用的缓存及初态组划分。读取新数据时分别覆盖 `task.dataset.dataset_path` 或 `task.dataset.prepared_dir`。Task1 official 的 proprio9 随 dataset_path 一起切换。

## 新增任务

1. 在 `tasks/taskN_name/sim` 实现观测、物理动作执行、任务判定和安全测量，在 `tasks/taskN_name/data` 实现数据准备。Task1 的物理代码仍在 `tasks/task1_dimm_insert/env`，因为正在运行的实验直接引用该包。
2. 在 `learning/dataset` 实现 `BaseImageDataset`。单样本返回 `{"obs": {key: Tensor[T_obs,...]}, "action": Tensor[horizon,D]}`；RGB 使用 CHW 浮点张量。实现 `get_validation_dataset()` 和 `get_normalizer()`；按完整回合或共享初态组划分，归一化只拟合训练数据。序列不得跨回合。
3. 在 `learning/env_runner` 实现 `BaseImageRunner`，构造函数接收 `output_dir`。`run(policy)` 负责复位、观测堆叠、`policy.predict_action(obs)`、动作执行和结果记录，返回至少 `{"test/mean_score": 任务成功率}`。安全指标独立返回。
4. 增加 `learning/config/task/taskN_image.yaml`，指定 shape_meta、crop_shape、dataset 和 env_runner 的 `_target_`。

配置结构：

```yaml
name: taskN_name
crop_shape: [76, 76]
shape_meta:
  obs:
    external: {shape: [3, 128, 128], type: rgb}
    state: {shape: [12], type: low_dim}
  action: {shape: [7]}
dataset:
  _target_: learning.dataset.taskN_image_dataset.TaskNImageDataset
  dataset_path: /绝对路径/TaskN数据目录
  horizon: ${horizon}
  n_obs_steps: ${n_obs_steps}
env_runner:
  _target_: learning.env_runner.taskN_image_runner.TaskNImageRunner
  n_test: 3
  test_start_seed: 10000
```

形状与类名按新任务填写。接入后仍使用 `train.py task=taskN_image` 与同一个 `eval.py`；公共训练入口和官方模型无需增加任务分支。

## 检查点导入

已有同结构官方 Hybrid 策略的 model/ema 权重可转成 Workspace 推理检查点：

```bash
algos/diffusion_policy/.venv/bin/python -B -m learning.import_checkpoint \
  --source /绝对路径/旧权重.pt --task task2_image \
  --output /绝对路径/新推理权重.ckpt
```

使用严格形状加载，保留权重中的归一化参数。该命令只导入推理权重，不恢复优化器或学习率进度。已有 Task2 epoch18 EMA 的统一格式权重位于：

`/mnt/data2/wjy_SERL/Embodied_Safety/outputs/checkpoints/task2_riser_io/official_dp_epoch018.ckpt`

## 已知限制

- Task1 正在运行的 3090 实验继续使用其原配置和生命周期脚本；其恢复/回传流程仍依赖 `tools/task1_official_compare_v001`。统一入口不接管在途进程。
- 上游 Workspace 的 resume 轮数与学习率调度按其自身实现处理。跨不同训练预算续训不能视为与原定完整预算等价；本项目尚未验证这类断点恢复的调度连续性。
- Task2 当前安全评估使用已冻结的螺丝装配经验边界；根部拉力和过弯仍作测量，实物损伤阈值未标定，损伤率为 null。安全完成率只适用于记录的评估范围。
- Task2 历史 DAgger 混合采集使用独立 PolicyClient 格式；Workspace 权重尚未接入该采集接口。部分专家标签的纠偏数据需标签掩码适配。
- 接口短测验证数据对齐、梯度更新、权重加载与物理执行；不能代替完整预算的策略成功率测试。
