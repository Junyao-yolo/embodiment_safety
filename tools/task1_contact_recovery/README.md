# Task1 接触纠偏

在 ufdwaa 的 `/mnt/data2/wjy_SERL/Embodied_Safety` 中执行。复用现有
`algos/diffusion_policy/.venv/bin/python`、RTX 4090 和项目依赖，不安装新环境。

## 流程

1. 等待官方输入、当前输入各 20 条原始测试完成，生命周期状态为 complete。
2. 按完整成功、插入准备、脱架条数依次选择本轮微调的父策略。
3. 从该策略的接触失败案例选择最多 10 个接管时点。学生使用相同种子重新执行，
   教师在插入入口、失败前或临近失败时接管。保存实际接管状态与原案例的差异；
   这不是仿真状态的精确快照恢复。
4. 教师复用 Task1Expert，通过同一物理速度控制器保持夹持、退出接触、对齐和插入。
   不直接改物体位置、扣具状态或成功判据。松手必须满足实测就位和扣合历史。
5. 每次教师最长 120 秒仿真时间；失败记录只用于诊断。至少 3 个无接管后安全告警的
   完成片段、且覆盖至少 2 个来源种子，才开始梯度更新。失败不能直接归为机械不可恢复。
6. 保留全部原始 96/11 划分，用 75% 原始窗口和 25% 教师窗口微调。
   同一来源种子的不同接管时点归入同一划分。归一化沿用父策略，EMA 初始化模型，
   新 AdamW，lr=1e-5、batch32、3 epochs，约 10,134 次更新。
7. 根目录 train.py 使用官方 Workspace 训练，按验证 loss 选择检查点。
   父策略与微调策略通过 eval.py 各测试 20 条，相同新种子 2026112001 起、执行块8，
   无教师。报告开扣回退、插入和完整成功率，不能把教师完成率当学生成功率。

采集和评估均保留视频、阶段、接触告警、仿真时间和墙钟时间。
输出目录：`outputs/runs/task1/diffusion_policy/contact_recovery_v001`。
当前正在进行的基线评估完成前，队列不启动仿真或 GPU 训练。

## 命令验证

以下两条命令不占用 GPU，不启动训练或仿真：

```bash
algos/diffusion_policy/.venv/bin/python -B -m tools.task1_contact_recovery.queue --describe
algos/diffusion_policy/.venv/bin/python -B -m pytest -q tools/task1_contact_recovery/test_recovery.py
```

预期分别打印流程参数、5 个测试通过。测试检查接管时点、就位前禁止松手、
超力片段排除及按来源种子划分数据；不证明教师已能物理纠偏。

## 启动

```bash
algos/diffusion_policy/.venv/bin/python -u -B -m tools.task1_contact_recovery.queue
```

启动器应以独立后台进程运行并记录日志。已有 selection.json 时禁止从头覆盖重跑。
首个真实教师接管是物理集成验证；异常时队列记录 failed 并停止，不进入训练。
