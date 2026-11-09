# Task1 采集优化结果

2026-11-04，主机 isaac-b。v005 保留 13 条，v006 保留 33 条，优化后已恢复 v006 的两个采集 worker。现有目标为 v006 达到 100 条；合并 v005 后为 113 条。

## 实测收益

同一批 120 个历史目标、初始关节角和关节边界，交替测试新旧求解器：

| 指标 | 原求解器 | 优化求解器 |
| --- | ---: | ---: |
| 总经过时间 | 20.734 秒 | 3.083 秒 |
| 平均残差评估次数 | 353.37 | 75.68 |
| 满足原位姿容差及关节限位 | 120/120 | 120/120 |
| 最大位置误差 | 2.430 微米 | 4.988 微米 |
| 最大姿态误差 | 0.0001283 弧度 | 0.0002000 弧度 |

该 IK 微基准耗时降低 85.1%，约 6.73 倍速度。解析雅可比与数值差分抽查 10 组，最大相对误差 3.415e-11。可达目标、不可达目标和关节边界检查通过。

相同随机种子的完整轨迹验证：

| 随机种子 | 历史总实际时间 | 优化后总实际时间 | 历史仿真时间 | 优化后仿真时间 | 任务/数据/平滑度验收 |
| --- | ---: | ---: | ---: | ---: | --- |
| 2026110721 | 25.27 分钟 | 11.32 分钟 | 89.3 秒 | 89.4 秒 | 通过 |
| 2026110722 | 25.34 分钟 | 11.00 分钟 | 91.1 秒 | 87.6 秒 | 通过 |

平均单条实际时间由 25.30 分钟降至 11.16 分钟。两条均有 accepted=true、data_valid=true、motion_quality.accepted=true，safety_pass=false。计时验证数据在 outputs/datasets/task1/parallel_speed_check_v001，随机种子与历史轨迹重复，不加入训练集合。

## 当前耗时构成

两条完整验证的录制阶段平均分项；下表采用互斥主机经过时间，未插入 GPU 同步：

| 分项 | 平均秒数 |
| --- | ---: |
| kit_native_update | 324.55 |
| contact_report | 97.59 |
| controller_tick | 88.33 |
| image_encode_write | 30.03 |
| physics_callback | 29.07 |
| physics_log_write | 28.57 |
| contact_metrics | 15.50 |
| finish_and_validation | 5.07 |
| camera_update | 4.63 |
| state_and_image_snapshot | 3.50 |
| transition_write | 1.51 |
| expert_action | 0.82 |

平均重置/准备 39.20 秒，录制 629.48 秒，结束处理 0.98 秒。
录制阶段的 IK 调用由历史平均 467.67 秒降至 88.33 秒。
原生仿真/渲染调用与接触回调是当前主要剩余开销。
kit_native_update 包含物理、渲染和驱动等待，不能等同于 GPU 核函数时间。
旧 ik_trace.jsonl 也包含准备阶段，本报告按录制观察时间范围筛选，避免把准备求解计入录制耗时。

## 执行改动

- 使用 Lula 解析雅可比，达到任务验收容差十分之一时结束局部有界求解。当前停止精度为 5 微米、0.0002 弧度；原 50 微米、0.002 弧度验收门限继续保留。
- 保留原关节限位、关节速度限制、控制周期、动作限幅和最大求解评估预算。
- 跳过已关闭辅助吸附对应的冗余位姿读取及几何计算。
- 原生日志启用异步写入，将逐条文件刷新阈值从 verbose 改为 error；所有警告仍保留。[NVIDIA 日志设置说明](https://docs.omniverse.nvidia.com/kit/docs/carbonite/latest/docs/logging/Logging.html)
- 三路 256×256 图像、240 Hz 物理、10 Hz 动作、相机参数、场景和示范随机化 profile 保持现有配置。
- 记录每条轨迹的分项时间与实际求解器名称，v006 混合历史/优化求解器时在清单中列明。
- 新相机合并清单指向 v005/v006 的成功 episode，训练配置使用该清单；100000 次更新与 batch=16 保留，计划 episode 划分为 102/11。未启动训练。

## 运行状态与预期

2026-11-04 23:04 CST：v006 为 33/100，两个 worker 正在处理 attempt_053/054，无控制失败标记。本次未停止或修改 Task2；23:04 复查时，此前的 Task2 进程已退出，GPU 计算进程只剩两个 Task1 worker。
单条按 11–12 分钟估计，剩余 67 条全部成功约需 6 小时。历史已完成 48 次尝试、33 次通过（68.75%）；按该通过率估计约 9 小时，暂给 8–10 小时窗口，即 11 月 5 日 07–09 点。优化后的长期通过率尚未测定。

采集目录：
- /mnt/data2/wjy_SERL/Embodied_Safety/outputs/datasets/task1/parallel_diverse_v005
- /mnt/data2/wjy_SERL/Embodied_Safety/outputs/datasets/task1/parallel_diverse_v006
- 合并清单：/mnt/data2/wjy_SERL/Embodied_Safety/outputs/datasets/task1/parallel_diverse_new_wrist_v001/manifest.json

## 已知问题

- 完整轨迹对比的背景负载不同：历史为两个 Task1 加两个 Task2 仿真进程，验证时为两个 Task1 加一个 Task2。因此 25.3→11.2 分钟是实际观察值，不能将全部下降归因于代码。IK 微基准采用相同目标和交替计时，仍是主机负载条件下的配对测试，并非独占 GPU 基准。
- 现行 accepted 条件不要求 safety_pass=true。现有 v005/v006 共 46 条的 safety_pass 均为 false；新旧对照轨迹也均有约束告警。部分轨迹存在超过 2 mm 的相对滑移和接触力超限。保留这些原始标签，成功计数不代表严格安全示范验收。
- PhysX 的重复材料索引警告仍存在；异步缓冲减轻日志写入阻塞，没有消除警告来源。未改变碰撞几何、物理求解参数或隐藏警告。
- 两条完整验证支持功能与本次耗时比较，不能证明整体成功率提升。

## 文件变更与计时记录

相对 /mnt/data2/wjy_SERL/Embodied_Safety：

| 文件 | 类型 | 内容 |
| --- | --- | --- |
| tools/task1_execution_v004/controller_batched_fd.py | 修改 | 提取可覆写的求解和雅可比方法，默认数值求解行为保留 |
| tools/task1_lab_parallel_v001/controller.py | 新增 | 解析雅可比与位姿条件停止 |
| tools/task1_lab_parallel_v001/worker.py | 修改 | 采集求解器、无效计算裁剪、日志缓冲和分项计时 |
| tools/task1_lab_parallel_v001/performance.py | 新增，计时 | 互斥主机耗时统计 |
| tools/task1_lab_parallel_v001/benchmark_ik.py | 新增，基准 | 真实历史目标配对求解比较 |
| tools/task1_lab_parallel_v001/collect.py | 修改 | 清单记录实际求解器 |
| tools/task1_lab_parallel_v001/merge_manifests.py | 新增 | 合并成功轨迹清单 |
| tools/task1_lab_parallel_v001/README.md | 新增 | 采集、计时和新相机数据说明 |
| configs/experiments/task1_diffusion_parallel_diverse.json | 修改 | 新相机数据源、输出目录与 113 条划分 |
| profile_output/task1_collection_opt_20261104/refresh_manifest.py | 新增 | 采集期间刷新合并清单，采集退出后结束 |
| profile_output/task1_collection_opt_20261104/ | 新增，证据 | 修改前文件、配对 IK 结果、完整轨迹比较、GPU 快照、运行状态与本报告 |

计时输出保留用于后续确认瓶颈；如不再需要，可移除 worker.py 中的 Performance 调用与 performance.py，不影响控制逻辑。
