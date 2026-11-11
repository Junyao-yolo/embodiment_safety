# Task1 两组输入的 DAgger 对比

在 ufdwaa 的 `/mnt/data2/wjy_SERL/Embodied_Safety` 执行，复用
`algos/diffusion_policy/.venv/bin/python`，不安装依赖。

两组父模型分别为 `official_workspace_eval.ckpt` 与 `current_workspace_eval.ckpt`，
位于 `outputs/runs/task1/diffusion_policy/official_input_compare_3090_v001`。
保留父模型及原有 20 条测试。

从两组都达到脱架阶段的共同来源种子中，按种子顺序选择 5 个，每个种子选择
2 个不同的稳定夹持接管点。已有插入准备事件时从插入附近选择，否则从脱架附近选择。
选择阶段入口和接近夹持、姿态或安全条件失效前的状态，使用各自策略重新执行。
记录实际接管状态与参考状态的差异；不是精确快照恢复。两组交替执行，每组 10 次。

这是单轮干预式 DAgger：学生产生访问状态，教师接管完成任务，再汇总教师标签训练。
教师结束后本回合结束，本轮不包含微调后再次收集的第二轮 DAgger。
教师通过原有物理控制器完成保持夹持、退出接触、重新对齐、插入、扣合确认和松手。
教师每次最多 120 秒仿真时间，只纳入通过完整任务验收且接管后无安全告警的片段。
两组均须至少 3 条合格片段、来自至少 2 个来源种子，才进入梯度更新。
若任一组不足，保存两组诊断结果并停止；不能将教师失败直接视为机械不可恢复。

每组使用自己的纠偏数据，保留原始 96/11 回合划分。来源种子整体划分训练与验证。
原示范窗口占 75%，教师窗口占 25%；两组接受条数、时长与独立窗口数单独报告，
不要求两组纠偏数据相同。EMA 初始化、父模型归一化、新 AdamW、lr=1e-5、batch32、
3 epochs、seed42，预期各 10,134 次更新，验证 loss 选模。
这是两套输入方案接受相同微调流程的比较，不是仅改变状态维数的消融。

正式测试包括两组各自父模型及微调模型，共 4 个检查点，每个 20 条，共 80 条。
统一种子从 2026112001 开始，执行块8、最多3000步、单条墙钟2700秒、无教师。
记录阶段成功率、完整成功率、安全记录、仿真时间、墙钟时间及视频。

仿真在 ufdwaa 的 GPU 空闲后串行启动，不停止其他 GPU 作业。梯度更新在 machine2
使用两组各两张 3090：official 为3、4号卡，current 为5、6号卡，启动前重新检查空闲。
DDP 同步梯度，每卡 batch16、全局 batch32；两组并行训练，使用官方 Policy 和任务 Dataset，
专用 DDP 循环保存 Workspace 格式的模型、EMA、优化器以及调度器/RNG 状态。
最小训练包包含一份共享原始缓存、状态动作、两组父模型及必要代码；纠偏通过验收后再传入。
训练结束回传 best/latest、日志，严格加载检查模型和 EMA 后清理本次独立远端目录。
返回 ufdwaa 使用统一 eval.py 做无教师测试。

## 无 GPU 检查

```bash
algos/diffusion_policy/.venv/bin/python -B -m tools.task1_contact_recovery.dual_queue --describe
algos/diffusion_policy/.venv/bin/python -B -m tools.task1_contact_recovery.dual_queue --plan-only
algos/diffusion_policy/.venv/bin/python -B -m pytest -q tools/task1_contact_recovery/test_dual.py tools/task1_contact_recovery/test_recovery.py
```

预期打印两组参数、各 10 个接管点、7 个测试通过。检查不执行仿真、不做梯度更新。
首个真实接管仍需物理集成验证，接口检查不代表教师恢复已经成功。

## 后台启动

```bash
algos/diffusion_policy/.venv/bin/python -u -B -m tools.task1_contact_recovery.dual_queue
```

输出在 `outputs/runs/task1/diffusion_policy/dagger_input_compare_v001`。
`status.json` 记录当前阶段；`teacher_results.json` 记录两组纠偏结果；
`collection_summary.json` 和各组 `data_counts.json` 记录数据数量；
`comparison.json` 记录四个检查点的测试结果。已有 `plan.json` 时不覆盖重跑。

## 3090 环境与双卡验证

远端目录 `/mnt/file4/junyao/task1_dagger_compare_20261111`，SSH 别名 `task1-3090`。
环境声明为 `env-spec.json`；使用目录内 `.venv`，Python3.12、PyTorch2.7 CUDA12.8。
数据和环境准备可在等待纠偏仿真时进行。梯度正式微调仍要求两组纠偏数据均通过验收。

在本地执行：

```bash
ssh -o ConnectTimeout=15 task1-3090 'cd /mnt/file4/junyao/task1_dagger_compare_20261111 && python3 -u tools/task1_contact_recovery/witness_3090.py'
```

命令等待环境和原始输入准备完成，再等待指定 GPU 空闲。两组分别执行两次真实 DDP 更新，
校验两副本参数片段一致、完整检查点保存加载、DDPM100 输出 `[1,8,7]`，成功后打印
两组 `passed: true` 并写 `doc_witness_passed.json`。验证使用原始示范，不计入 DAgger
训练结果；验证权重随后删除，父权重保持不变。
