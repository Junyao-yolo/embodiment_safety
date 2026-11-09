# 目录职责

| 路径 | 职责 |
| --- | --- |
| train.py | 所有任务共用的 Hydra / Diffusion Policy Workspace 训练入口 |
| eval.py | 从 Workspace 检查点恢复策略，调用对应 EnvRunner |
| learning/config | 算法配置与 task 配置；观测、动作形状及数据/环境类 |
| learning/dataset | 各任务的 BaseImageDataset、观测编码 |
| learning/env_runner | 各任务的 BaseImageRunner 与仿真调度 |
| learning/controller | 将策略输出转换为各任务控制接口，保存动作块队列 |
| learning/common | 推理进程通信、共享状态表示 |
| learning/tests | 数据划分、序列边界、图像/状态/动作对齐 |
| third_party/diffusion_policy | 固定版本的官方 Policy、Workspace、模型和工具 |
| algos/diffusion_policy/.venv | 项目学习环境 |
| algos/diffusion_policy/vendor | 指向 third_party/diffusion_policy 的链接 |
| tasks/<task>/scene | USD 场景与资产 |
| tasks/<task>/sim | 物理执行、观测、验收和安全测量 |
| tasks/<task>/data | 原始采集数据到训练数据的准备工具 |
| tasks/<task>/configs | 任务物理、运动及判定配置 |
| tasks/task1_dimm_insert/env | Task1 物理执行；在途实验直接引用，暂不改名 |
| tasks/task1_dimm_insert/teleop | Task1 人工采集与安全反馈 |
| tasks/task2_riser_io/sim/collect | Task2 专家/DART 采集脚本 |
| algos/serl、algos/dp_sac | SERL 和 residual SAC 算法 |
| runners | SERL 运行、数据检查与仍被实验使用的工具 |
| tools/task1_official_compare_v001 | 正在运行的 Task1 3090 对照实验 |
| tools/task1_execution_v004 | Task1 原生物理执行后端 |
| outputs/datasets | Task1 数据、划分、缓存 |
| outputs/task2_riser_io | Task2 示范、历史训练、评估和纠偏结果 |
| outputs/runs | 新训练日志、检查点、闭环测量和视频 |
| outputs/checkpoints | 已导入统一 Workspace 格式的模型权重 |
| outputs/archive/tools | 已结束的 Task1 实验工具 |
| outputs/maintenance | 框架接口验证与源码归档 |

Task1 的原生物理执行后端为 `tools/task1_execution_v004/actor.py`；通过 `tasks/task1_dimm_insert/env/workspace_worker.py` 接入统一学习进程。Task2 的物理执行后端为 `tasks/task2_riser_io/sim/run_sim.py`。

新增任务遵循 `learning/README.md` 的 Dataset、EnvRunner 与 task 配置接口。模型和训练循环共用官方实现。
