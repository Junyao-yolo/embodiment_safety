# Task2 专家采集、数据与实验

统一训练入口：`/mnt/data2/wjy_SERL/Embodied_Safety/train.py`，使用 `task=task2_image`。

统一评估入口：`/mnt/data2/wjy_SERL/Embodied_Safety/eval.py`。

使用说明：`/mnt/data2/wjy_SERL/Embodied_Safety/learning/README.md`。

采集脚本在本目录。示范、权重、评估和历史纠偏结果在 `/mnt/data2/wjy_SERL/Embodied_Safety/outputs/task2_riser_io`。

| 路径 | 用途 |
| --- | --- |
| outputs/task2_riser_io/data/opposed_v002/raw | 100 条成功示范，三路 RGB128、23 维状态、48 维绝对专家动作 |
| outputs/task2_riser_io/data/opposed_v002/prepared | 按初态组隔离的 91/9 划分与无损图像缓存 |
| outputs/task2_riser_io/runs | 已有训练配置、日志、检查点 |
| outputs/task2_riser_io/evaluations | 已有闭环测试测量、判定与视频 |
| outputs/task2_riser_io/reports | 数据和运行诊断报告 |
| outputs/task2_riser_io/distillation_v001 | 历史纠偏数据与当时的采集副本 |

专家/DART 采集通过 `collect.py`、`collect_persistent.py`、`collect_parallel.py` 执行，见 `COLLECTION.md`。物理步骤位于 `/mnt/data2/wjy_SERL/Embodied_Safety/tasks/task2_riser_io/scripts/run_dp_bc.py`，运动和物理配置统一读取 `/mnt/data2/wjy_SERL/Embodied_Safety/tasks/task2_riser_io/configs`。

新训练与评估结果写入项目 `outputs/runs/task2_riser_io/diffusion_policy`。本目录保留专家采集及混合采集所需模块。
