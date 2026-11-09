ToolHang 使用现有 algos/diffusion_policy/.venv/bin/python，PyTorch 2.7.0+cu128。独立依赖目录 tools/tool_hang_benchmark_v001/deps；不修改基础环境。声明见 tools/tool_hang_benchmark_v001/env-spec.json。工作目录 /mnt/data2/wjy_SERL/Embodied_Safety。验证命令：algos/diffusion_policy/.venv/bin/python -u tools/tool_hang_benchmark_v001/smoke.py。期望 TOOL_HANG_SMOKE_OK。GPU 为 RTX 4090，其他任务保留。
2026-11-07 独立文档执行验证：退出码 0，TOOL_HANG_SMOKE_OK；RTX 4090；动作 [1,8,10]；两路 RGB 均 [240,240,3]。

官方模型成功率对照：现有 venv，加进程局部 official_deps/robomimic 0.2.0；基础环境不变。模型及优化器已完成 batch=64 FP32 的 200+500 次更新。完整训练保存和环境推理验证命令：algos/diffusion_policy/.venv/bin/python -u tools/tool_hang_benchmark_v001/train.py --config outputs/runs/robomimic/tool_hang_official_success_v001/smoke_config.json。应退出 0，smoke/completed.json 中 step=2，smoke/eval_step_000002/summary.json 中 completed=1，steps<=8；此验证不用于成功率统计。
2026-11-07 官方模型独立执行验证通过：完整训练命令退出 0；step=2；独立加载 EMA 检查点并执行 8 个环境步，FP32、DDPM100；仅验证执行，不计入成功率。
