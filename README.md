# Embodied Safety

Franka 装配任务的仿真、数据采集与安全评估。项目根目录：

`/mnt/data2/wjy_SERL/Embodied_Safety`

Task1、Task2 与后续任务共用 Diffusion Policy 训练和评估框架。模型、Workspace、序列采样器和归一化器采用固定版本的官方实现，任务差异通过 Dataset、EnvRunner 和 Hydra task 配置接入。

- [Diffusion Policy 使用与新任务接入](learning/README.md)
- [Task1：DIMM 安装](tasks/task1_dimm_insert/README.md)
- [Task2：线缆接线与螺丝紧固](tasks/task2_riser_io/README.md)
- [目录职责](LAYOUT.md)
- [DP-BC + residual SAC](algos/dp_sac/README.md)

## 目录

训练和评估只从根目录 `train.py`、`eval.py` 进入。任务差别在 `learning/config/task`、`learning/dataset`、`learning/env_runner`。

```text
train.py / eval.py
learning/          配置、Dataset、EnvRunner、动作执行
tasks/<任务>/      scene 场景，sim 或 env 物理，data 准备
third_party/diffusion_policy/   官方 Policy 与 Workspace
outputs/           数据集、checkpoint、评估和历史结果
```

`learning/controller` 把策略输出变成各任务的机器人指令。官方网络仍在 `third_party/diffusion_policy`。`algos/diffusion_policy/vendor` 是指向官方源码的链接，供仍在运行的 Task1 对照实验使用。

`runners/`、`algos/serl`、`algos/dp_sac` 保留 SERL 与 residual SAC。正在运行的 3090 实验继续使用 `tools/task1_official_compare_v001` 和 `tools/task1_execution_v004`。已结束的 Task1 实验工具在 `outputs/archive/tools`。

## 训练与评估

在 ufdwaa 的项目根目录执行：

```bash
algos/diffusion_policy/.venv/bin/python -B train.py task=task1_image training.num_epochs=100
algos/diffusion_policy/.venv/bin/python -B train.py task=task2_image training.num_epochs=100
```

这两条命令分别启动训练；根据可用 GPU 选择执行。评估使用同一个入口，任务配置随检查点保存：

```bash
algos/diffusion_policy/.venv/bin/python -B eval.py \
  --checkpoint /绝对路径/checkpoints/latest.ckpt \
  --output-dir /mnt/data2/wjy_SERL/Embodied_Safety/outputs/runs/独立评估目录 \
  task.env_runner.n_test=20 n_action_steps=4
```

训练输出位于 `outputs/runs/<task.name>/diffusion_policy/<时间戳>`。大型 USD 依赖、原始图像、权重及视频保存在 ufdwaa，不提交到 Git。任务成功与安全完成分别统计。
