# Task2 单根线缆接线与螺丝紧固

Franka 抓取环形端子、搬运落座，再抓取螺丝并旋拧，松爪后检查装配保持。

场景：`/mnt/data2/wjy_SERL/Embodied_Safety/tasks/task2_riser_io/scene/riser_io_success_v014.usda`。

项目共用学习框架：`/mnt/data2/wjy_SERL/Embodied_Safety/learning/README.md`。

```bash
cd /mnt/data2/wjy_SERL/Embodied_Safety
algos/diffusion_policy/.venv/bin/python -B train.py task=task2_image training.num_epochs=100
```

| 路径 | 内容 |
| --- | --- |
| learning/config/task/task2_image.yaml（项目根目录下） | 观测、动作、数据与 EnvRunner 配置 |
| sim | 物理执行、测量、任务验收与安全判定 |
| sim/collect | 专家与 DART 采集脚本 |
| configs | motion.json、runtime.json、safety.json |
| data | 训练数据准备 |
| outputs/task2_riser_io/data | 原始示范、划分清单、图像缓存 |
| outputs/task2_riser_io/runs、evaluations | 已有权重、训练日志、测试测量与视频 |
| scripts | 场景构造和几何工具 |
| scene | USD 场景与资产 |
| archives | 源码恢复包 |

数据集：`/mnt/data2/wjy_SERL/Embodied_Safety/outputs/task2_riser_io/data/opposed_v002/raw`，100 条成功示范。专家采集说明：`/mnt/data2/wjy_SERL/Embodied_Safety/tasks/task2_riser_io/sim/collect/COLLECTION.md`。

任务完成与安全完成分开统计。任务验收使用 `task2_released_assembly_v004`，检查端子落座、螺丝装配及松爪后保持。策略评估同时要求螺纹啮合验证。经验安全边界位于 `/mnt/data2/wjy_SERL/Embodied_Safety/tasks/task2_riser_io/configs/safety.json`。损伤极限尚未做实物标定。

新训练与评估输出到项目 `outputs/runs/task2_riser_io/diffusion_policy`。四螺丝布局和交互面板文档描述历史场景，当前训练使用本页所列 USD 与统一学习配置。
