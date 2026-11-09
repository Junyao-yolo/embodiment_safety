# Task1：DIMM 安装

所有命令在远端执行：
```bash
ssh isaac-b "cd /mnt/data2/wjy_SERL/Embodied_Safety && ..."
```


## Diffusion Policy

Task1 与 Task2 共用项目级训练、评估和官方 Workspace。使用说明：

`/mnt/data2/wjy_SERL/Embodied_Safety/learning/README.md`

在项目根目录执行：

```bash
algos/diffusion_policy/.venv/bin/python -B train.py task=task1_image training.num_epochs=100
```

配置 `task1_image` 使用三路 RGB256、两帧观测、state33 与 7 维动作；`task1_official_image` 使用外部/右腕两路 RGB240 与实测 9 维本体状态。107 条示范按完整回合划分为 96 条训练、11 条验证。

当前数据：`/mnt/data2/wjy_SERL/Embodied_Safety/outputs/datasets/task1/diffusion_parallel_diverse_new_wrist_v001`。

任务配置：`/mnt/data2/wjy_SERL/Embodied_Safety/learning/config/task/task1_image.yaml`。

物理执行在 `env/`，人工采集在 `teleop/`，场景在 `scene/`。正在运行的 3090 实验直接引用 `env/`，这次不改名。动作执行适配在 `learning/controller/task1_controller.py`。

新训练输出：`/mnt/data2/wjy_SERL/Embodied_Safety/outputs/runs/task1_dimm_insert/diffusion_policy`。

## SERL 环境与实验

下述 state31、单帧接口和历史数据量属于 SERL；Diffusion Policy 输入编码与当前数据划分见上面的配置。

## 当前接口

- 场景：`tasks/task1_dimm_insert/scene/collection.usda`。与已确认的采集场景保持相同内容，底层 USD/贴图为远端只读引用。
- 策略：external、wrist_left、wrist_right 三路 uint8 RGB，每路 256×256，T=1。
- 状态 31D：9 关节位置 + 9 关节速度 + 实测夹爪宽度 + 3 目标位置 + 4 目标四元数 + 夹爪目标，共 27D，再加 4 个阶段已获奖标记。
- 不把 socket/DIMM 真值、卡扣角度、接触力、刚度直接送入策略；这些用于环境奖励及诊断。
- 动作 7D，范围 [-1,1]：末端 xyz 线速度、xyz 角速度、夹爪速度。
- 单位缩放：0.32 m/s、1.57079632679 rad/s、0.06 m/s；最大目标超前 5 mm。不是关节速度“用米”。
- 每动作保持 100 ms，对应 24 个 240 Hz 物理步。中断不补齐、不补造下一观测。
- 真终止 mask=0，时间截断 mask=1；传感失效不作为有效训练样本。
- 每回合清空奖励标记、终止历史与启动辅助释放历史。

## 奖励

实现：`env/reward.py`；条件：`env/milestones.py`；终止：`env/termination.py`。
奖励版本 `task1_stage_once_v001`，接口 `task1_rgb_state31_reward_history_v001`。

| 阶段 | 每回合最多一次 |
|---|---:|
| 两侧开扣且撤离后路径可用 | 0.05 |
| 许可区抓稳并脱离置物架 | 0.10 |
| 插入准备位与姿态对正 | 0.05 |
| 入座与双侧锁止 | 0.20 |
| 松夹撤离并保持 | 1.00 |

最大总和 1.40。这是一次性阶段奖励，不是势函数塑形。
当前 0.20 入座奖励保守地在最终验收边界发放。
终局使用已接受场景上的操作判据：开到闭历史、落座、支撑接触、启动辅助释放、
松夹、撤离与稳定保持；不是额外的材料强度认证。
诊断力阈值：两侧按钮各3N，DIMM环境5N，两指各5N；不称厂家额定值。
启动闭位辅助在开扣接触/转动后单向释放，不在入座时重新开启；关闭 XY 吸附。

## 实际训练数据

`configs/data_manifest.json` 给出本机路径。
- 当前可直接训练：1个完整回合的1349条转移，加493条严格对齐的历史片段，共1842条。完整回合回报1.40。
- 原始1350条中的最后1条已位于因果成功终止之后，索引不纳入，原文件不改。
- 迁入28个有效在线诊断转移，不含人为注入急停样本。
- 历史15条标签已迁入 outputs/datasets/task1/aligned_history/historical_labels。03–15去重；01/02/03分别选入163/163/167条。\n- 50ms相邻行或100ms内子步只有动作一致、24条指令连续、位置/旋转/夹爪目标逐步积分一致、RGB有效时才接入；其余不平均、不插值伪造。\n- 完整回合因果计算阶段标记，再选取片段；不在片段开头清空奖励历史。片段不冒充新增完整成功回合。
- 没有独立留出完整评估回合；不能声称“15条独立示范已正式训练”。
- 大图像和原始记录仍位于远端 franka_sim，只读引用，文件哈希在输出 manifest 内。

## 运行

离线检查解释器：
`/mnt/data2/wjy_SERL/Embodied_Safety/tasks/task2_riser_io/.print_venv/bin/python`。
学习器解释器：
`/home/ufdwaa/anaconda3/envs/env_isaaclab/bin/python`。
Isaac：`/mnt/data2/isaacsim-5.0.0/python.sh`。
外部目录只提供已有第三方依赖，SERL 项目源码使用本仓库 algos/serl/vendor。

在仓库根目录执行：
```bash
# 数据准备：仅首次，输出已存在时拒绝覆盖
<离线Python> -B -m runners.prepare_task1
<离线Python> -B -m runners.preflight
<离线Python> -B -m unittest discover -s tests

# 固定数据混合更新；默认1:1示范/在线，batch=2、UTD=1，为小预算起始配置
<学习器Python> -B -m runners.train --updates 100 --output outputs/runs/task1/serl/first
# 续训：恢复优化器、参数和采样器状态
<学习器Python> -B -m runners.train --updates 100 --checkpoint outputs/runs/task1/serl/first/checkpoint.msgpack --output outputs/runs/task1/serl/continued

# 短回合接口验证（不是任务成功评估；动作限为2mm/s，旋转和夹爪禁用）
python3 -B -m runners.rollout --smoke --checkpoint <checkpoint.msgpack> --output outputs/runs/task1/serl/smoke

# 完整原生时钟回合；完整7D动作、真实接触，--learn每10个动作做混合更新
python3 -B -m runners.rollout --learn --checkpoint <checkpoint.msgpack> --output outputs/runs/task1/serl/episode
# 独立评估：确定性策略、不开学习、数据不混进训练replay
python3 -B -m runners.eval --checkpoint <checkpoint.msgpack> --output outputs/runs/task1/serl/eval
```

每个输出目录必须为新目录，避免覆盖。完整回合最多240秒仿真时间。
每次命令启动新的Isaac进程，关闭后下一次从原始场景复位。
人工启动：Isaac打开 scene/collection.usda，Script Editor运行 teleop/launch_collect.py。
不要同时在同一进程载入旧 franka_sim 包与新仓库包。

固定数据更新成功不等于完整策略任务成功；查看 configs/validation.json 的实际测试范围。

历史对齐回归已通过22项测试，并在 outputs/runs/task1/serl/history_aligned_check 完成2次实际混合更新；日志记录抽到02和03的历史片段。新数据索引有独立哈希，旧索引的checkpoint不会被静默续训到新数据集。

当前回归测试共27项。控制器停止时保存停止前IK残差、候选关节限位余量和目标/实测位姿；真实失败末段按已观测物理步写入终止转移（mask=0），不把已发出但尚未执行的命令计入100ms边界。验证结果见 outputs/runs/task1/serl/aligned_budget100_eval_diagnosed/diagnosis.json；该次策略未完成任务，不代表训练已收敛。

连续训练批次：python3 -B -m runners.batch --checkpoint <checkpoint.msgpack> --output outputs/runs/task1/serl/<new_batch>。最长1小时，最后预留600秒独立评估。每回合继承上次检查点，使用不同策略采样种子，以新的Isaac进程复位。程序/数据异常停止采样；单回合墙钟超过600秒标记未完成。summary.json 区分在线采样、更新与隔离评估；未完成的回合可能已写入部分有效转移，不能把完整回合计数当作replay总量。默认场景最大时限仍为240秒仿真时间，奖励和动作尺度不变。

首个有界批次 hour_batch_001 用时49.06分钟：35训练回合、1隔离评估，新增1145条训练转移、98次更新，累计204次更新。所有回合阶段奖励为0、完整成功0次，均在IK检查处提前终止。acceptance.json 已核对逐条哈希与终止标签；不能把训练链路通过解读为策略学会任务。该批次旧checkpoint.json中的updates_this_run、online_transitions和scope继承了上游元数据；本批次真实统计以worker_result.json/acceptance.json为准，后续写入已修正，旧产物不静默改写。


BC 训练入口：使用学习器Python运行 -B -m runners.pretrain_bc --updates 6000 --batch-size 16 --min-updates 2000 --patience 10 --eval-every 100 --output outputs/runs/task1/serl/<new_bc_run>。
这是独立BC初始化实验，不覆盖直接SAC检查点。均方误差按平移、旋转、夹爪三组等权；半数均匀抽样，半数按动作类型均衡抽样。
按每100步时间块留出80..89步检查、75..79及90..94步清除；这是相关轨迹内的检查，不代表独立任务泛化。
BC更新Actor和图像编码器；Critic未训练，随机标准差未标定。转入普通SAC结构后推理必须一致。
当前BC检查点只允许确定性评估，未开启SAC微调。记录实际更新次数、样本覆盖率和预算/平台期停止原因。
评估入口：python3 -B -m runners.eval --checkpoint <bc_checkpoint> --output outputs/runs/task1/serl/<new_eval>；默认完整240秒上限。
离线回放：使用离线Python -B -m runners.render_rollout --run <eval目录>，只合成已保存RGB，不启动仿真。

评估更正：vendor sample_actions 的确定性参数为 argmax=True，且不传seed。旧deterministic=True参数被忽略，
因此hour_batch_001/evaluation实际仍为随机采样；见EVALUATION_MODE_CORRECTION.json。旧原始记录保留。
新动作日志明确记录sampling_mode。BC 400步只是一轮试跑，不作为最终训练效果。

完整BC批次 bc_full_001：4800次更新后按检查集平台期停止，选中4300次检查点；1482条训练样本全部覆盖，检查样本未参与梯度。完整回合 bc_full_001_final_eval 在40.18秒因dimm_environment_N=6.435N超过5N停止，阶段奖励0、姿态相对初始最大偏转90.33度。手掌碰撞体接触PCB，两指接触力为0。任务未通过，未启动SAC微调。原SAC正确确定性对照在2.10秒IK停止、最大偏转142.79度。两者MP4及GIF均在对应输出目录，不改源场景。

### Full historical BC pair training

All 15 historical recording directories are included in the BC-only pair dataset, plus the native demonstration. Repeated motion groups are retained with their actual images, not claimed as independent trajectories. Dataset: outputs/datasets/task1/bc_all_demos_001 (20618 pairs). Historical 03-15 contribute all 1228 rows each; 01/02 retain 1638/1667 rows, rejecting 2255 stale/future RGB observations in total.

BC targets are the actual first substep velocity command at the observed boundary, never the averaged 100 ms action. Historical source commands last 50 ms; native commands last 100 ms. Deployment remains 100 ms, requiring closed-loop evaluation. These pairs have no reward/next-state training targets and MUST NOT enter SAC replay. Historical progress inputs use pre-action shadow history. No scene or control thresholds changed.

Prepare: python -m runners.prepare_bc --output outputs/datasets/task1/NEW_DATASET
Train: python -m runners.pretrain_bc --dataset outputs/datasets/task1/bc_all_demos_001 --output outputs/runs/task1/serl/NEW_RUN --updates 20000 --min-updates 10000 --patience 10 --eval-every 500 --batch-size 16 --cache-size 512

Same elapsed-time blocks are assigned together across repeated trajectories. This is correlated temporal validation, not independent task generalization. Outputs for this run: outputs/runs/task1/serl/bc_all_demos_001/fit.
