# Diffusion Policy 依赖与学习环境

统一训练入口：`/mnt/data2/wjy_SERL/Embodied_Safety/train.py`。

统一评估入口：`/mnt/data2/wjy_SERL/Embodied_Safety/eval.py`。

任务配置、数据适配和使用说明：`/mnt/data2/wjy_SERL/Embodied_Safety/learning/README.md`。

官方源码位于 `/mnt/data2/wjy_SERL/Embodied_Safety/third_party/diffusion_policy`，仓库 https://github.com/real-stanford/diffusion_policy ，版本 `5ba07ac6661db573af695b419a7947ecb704690f`，MIT License。本目录的 `vendor` 是指向该目录的链接。公共框架直接调用：

- `DiffusionUnetHybridImagePolicy`
- `TrainDiffusionUnetHybridWorkspace`
- `BaseImageDataset`、`ReplayBuffer`、`SequenceSampler`
- `BaseImageRunner`、`LinearNormalizer`

`.venv/bin/python` 为训练与推理解释器。依赖路径由 `learning/bootstrap.py` 配置；Isaac 使用独立解释器，通过本机进程通信接收策略动作。

本目录中仍被已有实验调用的数据表示、策略和执行辅助模块用于相应模型接口。新增 Diffusion Policy 训练使用项目根目录入口。
