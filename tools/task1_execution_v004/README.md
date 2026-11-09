# Task1 固定成功回放执行基准

正式交付候选为 batched_io，保留原渲染和全部控制精度。

代码只在本目录加载，不替换公共控制器或场景。动作源、初态、阈值、物理频率和对照容差见 `contract.json`。

## 运行

在 isaac-b：

```bash
cd /mnt/data2/wjy_SERL/Embodied_Safety
python3 -B tools/task1_execution_v004/run.py --variant baseline --steps 60 --name baseline_check
python3 -B tools/task1_execution_v004/run.py --variant batched_io --steps 60 --name optimized_check
```

切回基线使用同一入口的 `--variant baseline`。每次使用新的产物名称。

从相同初态执行带接触窗口验收门的回放：

```bash
TASK1_WINDOW_GATES=1 python3 -B tools/task1_execution_v004/run.py --variant batched_io --steps 1349 --name optimized_full_check
```

窗口分别位于第 50、490、870、1100 个动作后。对照失败立即停止。只有完整五阶段、数据对齐、安全与接触审计通过，才能进入同检查点 DP 测试。

## 计时口径

`cycle_timing.jsonl` 按 100 ms 观测边界记录互斥 CPU 墙钟时间；第 0 行是启动阶段，性能统计从第 1 行开始。每周期保存模块耗时和次数。嵌套模块不重复相加。验收门运行时间单列 `validation_gate`。

`ik_calls.jsonl` 保存实际 IK 调用、残差、雅可比、TRF 迭代、重试和最终误差。每个 100 ms 动作对应两次 50 ms 控制器更新；两次目标不同，均保留。初值使用上一有效解，首次使用实测关节状态。

GPU 图形队列与 PhysX 的独立执行时长尚未拆分。相机读取耗时不等于 GPU 渲染耗时；`kit_native_unresolved` 保留为原生 Kit、图形和驱动等待的混合时间，不标记为纯 PhysX。没有插入额外 GPU 同步。

## 文件

- `actor.py`：隔离回放入口和窗口验收。
- `controller_baseline.py`、`execution.py`：固定回放控制和启动保持配置。
- `controller_cached_fd.py`：保留有限差分和 TRF，仅复用同次求解内的模型句柄、坐标变换与目标逆旋转，缓存完全相同输入的残差。
- `controller_analytic.py`、`kinematics.py`：解析雅可比对照候选。
- `solver_metrics.py`：返回 TRF 已有迭代计数，不改变迭代计算。
- `timing.py`、`summarize.py`：互斥计时及统计。
- `compare.py`、`window_gate.py`：按固定容差对齐比较。
- `audit_kinematics.py`：真实姿态上的残差／雅可比数值核对。
- `audit_episode.py`：转移、图像对应及实测接触审计。
- `progress.py`：成功回放与 DP 的连续进展、动作跟随和接触峰值比较。

产物位置：`outputs/runs/task1/diffusion_policy/performance_retest_v004/`。

## 批量有限差分候选

`batched_fast` 保留原 TRF 与全部有限差分采样点，批量执行旋转残差；同时启用额外视口关闭、相同 drive 值去重、NPZ 无压缩和 2 ms IPC 轮询。三路相机、物理时序、精度和动作不变。

```bash
python3 -B tools/task1_execution_v004/run.py --variant batched_fast --steps 60 --name batch_short_new
TASK1_WINDOW_GATES=1 python3 -B tools/task1_execution_v004/run.py --variant batched_fast --steps 1349 --name batch_full_new
```

`audit_batched_fd.py` 在安装的 Isaac Python 下独立核对雅可比和完整求解，不启动 SimulationApp。`native_profile.py` 仅在 `TASK1_NATIVE_PROFILE=1` 时启用原生短剖析；正式性能对照不启用。CPU 事件采用官方 10 ns 起始刻度，GPU 样本异步，详见 `analyze_native_profile.py`。

报告：`outputs/runs/task1/diffusion_policy/performance_retest_v004/REPORT.md`。`build_report.py` 只从现有结果更新报告。

## 观测一致的候选

正式回归候选使用 batched_io：原视口保持启用，批量有限差分及其余 I/O 优化保留。check_cameras.py 按 contract.json 中预先固定的 0.5 灰度级 MAE 容差检查。controlled_pipeline.py 串行执行短窗口、完整回放，再在验收通过后运行同一 DP 检查点。关闭默认视口的 batched_fast 只保留为性能对照，其腕部图像发生了可测变化。
