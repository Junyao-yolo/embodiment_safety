# Task2 专家示范采集

代码：`/mnt/data2/wjy_SERL/Embodied_Safety/tasks/task2_riser_io/sim/collect`。场景：`/mnt/data2/wjy_SERL/Embodied_Safety/tasks/task2_riser_io/scene/riser_io_success_v014.usda`。采集结果写到 `/mnt/data2/wjy_SERL/Embodied_Safety/outputs/task2_riser_io`。

## 执行与数据

`collect.py` 启动 `collect_persistent.py`，默认使用 `feedback_dart.json`。一个批次加载一次 Isaac 应用与场景；回合之间停止时间线，恢复物体、关节锚点和机器人初态，复用相机与渲染产品。

仅最终装配成功的完整轨迹进入 raw 数据目录。失败尝试的连续数值保存在 attempt_results；成功判定不因短时力值超限而拒收。

| 文件 | 内容 |
| --- | --- |
| state.npy | 23 维实际机器人状态；上一目标取实际执行值 |
| action.npy | 专家标签，六条绝对关节/夹爪指令组成 48 维动作 |
| executed_action.npy | 实际执行的含扰动动作 |
| action_source.npy | 各指令来源 |
| external/wrist_left/wrist_right.npy | 10 Hz 的三路 RGB128 |
| time.npy、episode.json | 时间、初态、路径族及标签语义 |
| diagnostics.json | 数值质量与连续测量位置 |

物理 120 Hz，指令 60 Hz，观测 10 Hz。当前 `outputs/task2_riser_io/data/opposed_v002/raw` 已有 100 条成功轨迹；训练窗口与划分由 learning Dataset 负责。

## 随机化、反馈和运动

线缆及近端附件整体移动以保持连接；目标板和支撑共同移动；螺丝在托架内扰动后随托架移动。机器人起始位姿、可抓取区域内的抓取点也随机化。相邻四个种子共享初态，分别使用 direct、left、right、high 路径族。

专家根据当前物体与机器人状态计算纠正指令，抓取后使用实测抓取变换。DART-style 采集向实际执行加入相关速度扰动，默认标准差 2 mm/s、相关时间 0.35 s，接触阶段幅度减为十分之一；标签在扰动前记录。该专家需要任务参考，不具备任意散乱状态下的全局重规划保证。

普通移动规划峰值 1000 mm/s、线加速度上限 2000 mm/s²；下降、落座、螺纹入口和旋拧分别减速，详见 `/mnt/data2/wjy_SERL/Embodied_Safety/tasks/task2_riser_io/configs/motion.json`。关节命令限制为 1.8 rad/s、10 rad/s²。速度指仿真时间，墙钟时长另行统计。

腕部相机分别位于手部 X=−80 mm、+80 mm 两侧，斜看指尖接触区域，光轴夹角约 93.7°。历史数据保留各自的相机位姿。

## 验收与安全

`task2_released_assembly_v004` 检查松爪后端子仍在目标座容纳范围内、螺丝穿入螺母、螺丝头和端子接触及装配保持。座和端子尺寸来自 USD。压紧力不设最低数值门槛。

力、力矩、速度、加速度、关节间隙和螺纹拟合继续记录。无法继续的非有限物理状态/动作及最终装配未完成属于失败。采集保留数值诊断；闭环策略评估的经验安全边界另见 `/mnt/data2/wjy_SERL/Embodied_Safety/tasks/task2_riser_io/configs/safety.json`。经验边界不等于材料损伤极限。

## 使用

单进程追加示范：

```bash
cd /mnt/data2/wjy_SERL/Embodied_Safety/tasks/task2_riser_io/sim/collect
python3 -B collect.py --collection-config feedback_expert.json --additional-successes 5
```

并行采集：

```bash
cd /mnt/data2/wjy_SERL/Embodied_Safety/tasks/task2_riser_io/sim/collect
python3 -B collect_parallel.py --output-root /mnt/data2/wjy_SERL/Embodied_Safety/outputs/task2_riser_io/data/opposed_v002 --workers 2 --successes 100 --seed 100
```

该命令在成功数量达到 100 时不追加已有回合。每个进程复用自己的场景；进度、日志和重置请求带 worker 后缀。已有成功与失败尝试的种子会跳过。

训练入口：

```bash
/mnt/data2/wjy_SERL/Embodied_Safety/algos/diffusion_policy/.venv/bin/python -B /mnt/data2/wjy_SERL/Embodied_Safety/train.py task=task2_image training.num_epochs=100
```

读取新增数据前，在项目根目录使用学习解释器运行 `-m tasks.task2_riser_io.data.prepare_diffusion --source <原始回合目录> --output <新准备目录>`，再通过 `task.dataset.prepared_dir` 指定。统一框架说明：`/mnt/data2/wjy_SERL/Embodied_Safety/learning/README.md`。

成功视频导出：

```bash
cd /mnt/data2/wjy_SERL/Embodied_Safety/tasks/task2_riser_io/sim/collect
/mnt/data2/wjy_SERL/Embodied_Safety/algos/diffusion_policy/.venv/bin/python -B export_success_video.py /mnt/data2/wjy_SERL/Embodied_Safety/outputs/task2_riser_io/data/opposed_v002/raw/episode_0100
```

## 已知限制

旧混合策略采集使用旧 PolicyClient 与模型格式；当前 Workspace 检查点尚未接入该接口。历史纠偏数据保存在 `outputs/task2_riser_io/distillation_v001`，带部分专家标签的数据需要相应掩码适配，不能直接作为完整标签训练。

CPU PhysX 与 GPU PhysX 的力值等价性尚未验证。当前配置使用 CPU PhysX 与 GPU 渲染，刚体求解迭代 128/8、关节系统 64/8。接触日志存在 getMaterialFromInternalFaceIndex 警告，测量仍需结合接触几何与实物标定解释。
