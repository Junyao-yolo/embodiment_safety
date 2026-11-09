# ToolHang DP-BC

使用仓库现有 Diffusion Policy 模型与 Python/PyTorch/CUDA 环境，对官方 ToolHang PH 的 200 条图像示范训练。实验依赖在本目录的 `deps/` 中，仅由本实验入口加入模块搜索路径。

工作目录：`/mnt/data2/wjy_SERL/Embodied_Safety`

环境与 CUDA、EGL 验证命令：

```bash
algos/diffusion_policy/.venv/bin/python -u tools/tool_hang_benchmark_v001/smoke.py
```

成功标记：`TOOL_HANG_SMOKE_OK`。该检查运行 CUDA 前向、反向、优化器更新和扩散推理，并在 ToolHang 中执行动作、读取两路 240×240 图像。随机张量更新耗时只用于预估，不代表真实数据吞吐量。

流程入口：

```bash
algos/diffusion_policy/.venv/bin/python -u tools/tool_hang_benchmark_v001/pipeline.py
```

流程等待 `download.py` 下载完成，然后准备官方 JPEG2000 缓存、校验三个示范终态和一次真实数据 CUDA 更新，再训练 100,000 次更新。全部 200 条示范参与按轨迹划分：196 条训练、4 条验证。训练结束及每 25,000 步均在 20 个独立随机初始状态上评估并保存视频。

模型沿用当前 Task1 的 ResNet18→128 维特征和 U-Net `[128,256,512]`，两帧观测、预测 16 步、执行 8 步，DDIM 16 次采样，x0 目标、BF16、batch=16、AdamW 1e-4、EMA。输入采用 ToolHang 的两路 240×240 RGB 和 9 维本体状态，绝对动作旋转转为 6D 后为 10 维。该实验检验当前模型实现，不是 Diffusion Policy 论文大模型配置的严格复现，也不载入 Task1 权重。

输出：`outputs/runs/robomimic/tool_hang_dp_bc_v001/`

- `pipeline_status.json`：数据准备、训练和评估流程状态。
- `progress.json`、`metrics.jsonl`：实际更新次数、loss、更新耗时及剩余训练时间估计。
- `timing.json`：真实数据前 200 次预热后，500 次更新计时；包含数据等待和训练更新，验证、保存和评估另计。其他 GPU 任务继续运行，所以不是独占性能测试。
- `checkpoints/last.pt`、`best.pt`：模型、EMA、优化器、学习率调度、随机数及训练位置的完整检查点。
- `eval_step_*/summary.json`：完整任务与装架、挂工具阶段成功率，以及推理、仿真和总耗时。
- `eval_step_*/episode_*.mp4`：20 Hz 仿真时间视频，两路相机并排；视频时长不等于推理墙钟时长。

训练完成与任务学会分别判断：梯度更新完成不代表完整任务成功，最终以独立状态评估的完整任务成功率为准。
