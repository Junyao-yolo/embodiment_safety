"""Build measured replay timing and policy-progress comparison artifacts."""
from pathlib import Path
import json,csv,subprocess
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT=Path("/mnt/data2/wjy_SERL/Embodied_Safety")
D=ROOT/"tools/task1_execution_v004"
R=ROOT/"outputs/runs/task1/diffusion_policy/performance_retest_v004"
def read(p):return json.loads(p.read_text())
names=["baseline_short_r2","analytic_short","fast_short","batched_fast_short","cached_fd_full","batched_fast_full","batched_io_short","batched_io_full","paired_baseline_short","paired_optimized_short"]
records=[]
for name in names:
    p=R/name/"timing_summary.json"
    if not p.exists():continue
    s=read(p)
    row={"run":name,"cycles":s["cycles"],"active_wall_s":s["active_wall_s"],
        "median_s":s["cycle_median_s"],"p95_s":s["cycle_p95_s"],
        "ik_s":s["ik"]["seconds"],"ik_calls":s["ik"]["calls"],
        "residual_evaluations":s["ik"]["residual_calls"],"iterations":s["ik"].get("iterations"),
        "native_kit_s":s["modules"].get("kit_native_unresolved",{}).get("total_s"),
        "wall_over_sim":s.get("wall_over_sim")}
    calls=[json.loads(x) for x in (R/name/"ik_calls.jsonl").read_text().splitlines()]
    nfev=sum(x["nfev"] for x in calls) if all("nfev" in x for x in calls) else None
    row["trf_nfev"]=nfev
    row["residual_python_calls"]=nfev if name.startswith("batched") or name=="paired_optimized_short" else s["ik"]["residual_calls"]
    records.append(row)
with (R/"timing_comparison.csv").open("w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
fig,ax=plt.subplots(figsize=(10,4.8))
short=[x for x in records if x["cycles"]==60]
labels=[x["run"].replace("_short_r2","").replace("_short","") for x in short]
bottom=np.zeros(len(short))
for key,label,color in [("ik_s","IK","#377eb8"),("native_kit_s","Native Kit / waits","#ff7f00"),("other","Callbacks / I/O / IPC","#999999")]:
    v=np.array([x["active_wall_s"]-x["ik_s"]-x["native_kit_s"] if key=="other" else x[key] for x in short])
    ax.bar(labels,v,bottom=bottom,label=label,color=color);bottom+=v
ax.set_ylabel("Wall seconds for identical 60 actions");ax.legend()
ax.set_title("100 ms control, 240 Hz physics; shared GPU")
fig.tight_layout();fig.savefig(R/"short_timing_comparison.png",dpi=160);plt.close(fig)
fig,axes=plt.subplots(2,3,figsize=(15,7),sharex=True)
series=[("success_reference","Successful action replay"),("full_episode_001","DP v003 episode 1"),("full_episode_002","DP v003 episode 2")]
if (R/"dp_episode_003_progress.csv").exists():series.append(("dp_episode_003","Current DP / original rendering"))
fields=[("stage_progress","Completed stages"),("target_distance_mm","DIMM target distance [mm]"),("orientation_deg","DIMM orientation error [deg]"),("actual_translation_mm_s","Actual EE speed [mm/s]"),("controller_goal_error_mm","EE goal lead [mm]"),("rise_mm","DIMM rise from rack [mm]")]
for name,label in series:
    with (R/(name+"_progress.csv")).open() as f:rows=list(csv.DictReader(f))
    t=np.array([float(x["time_s"]) for x in rows])
    for ax,(key,title) in zip(axes.ravel(),fields):
        y=np.array([float(x[key]) if x[key] else float("nan") for x in rows]);ax.plot(t,y,label=label,linewidth=1)
        ax.set_title(title);ax.grid(alpha=.2)
for ax in axes[-1]:ax.set_xlabel("Simulation seconds")
axes[0,0].legend(fontsize=8)
fig.tight_layout();fig.savefig(R/"policy_progress_comparison.png",dpi=160);plt.close(fig)
dp_rows=[]
for name,role in [("dp_episode_001","人为中断，不计策略成败"),("dp_episode_002","关闭默认视口的诊断"),("dp_episode_003","保留原渲染的单轮验收")]:
    p=R/name/"status.json"
    if p.exists():
        x=read(p)
        dp_rows.append(dict(name=name,role=role,status=x.get("status"),wall_s=x.get("wall_s"),steps=x.get("full_transitions"),reward=x.get("reward"),stages=x.get("stages"),video=x.get("video"),result=x.get("result")))
(R/"dp_test_summary.json").write_text(json.dumps(dp_rows,ensure_ascii=False,indent=2))
dp_table="\n".join("|"+str(x["name"])+"|"+str(x["role"])+"|"+str(x["status"])+"|"+str(x["steps"])+"|"+str(x["reward"])+"|" for x in dp_rows)
status=read(R/"controlled_workflow_status.json") if (R/"controlled_workflow_status.json").exists() else {}
native=read(R/"native_profile_summary.json")
batch=read(R/"batched_fd_audit.json")
acc=read(R/"controlled_execution_acceptance.json") if (R/"controlled_execution_acceptance.json").exists() else None
table="\n".join("|{run}|{cycles}|{active_wall_s:.2f}|{median_s:.3f}|{p95_s:.3f}|{ik_s:.2f}|".format(**x) for x in records)
text=f"""# Task1 固定成功回放与策略进展

保留原有限差分的完整回放已通过全部五阶段；1,350 个观测的末端、关节、DIMM 和卡扣数值与成功基准完全一致。批量有限差分的短窗口也完全一致。保留原渲染的 batched_io 候选当前状态：{status.get("phase","unknown")}。

## 基准与不变量

成功回放：`outputs/runs/task1/diffusion_policy/insertion_diagnosis_v001/recorded_full/eval/episode_01/`。
入口：`outputs/runs/task1/diffusion_policy/insertion_diagnosis_v001/recorded_full/pipeline.py`。
动作源：`/mnt/data2/wjy_SERL/franka_sim/outputs/datasets/task1_dimm_insert/reports/serl_collection_reproduce_v001/policy_reset_pair_002/cycle_01/episodes/20261030T142631_ca340bad/transitions.jsonl`。

共 1,349 个 100 ms 动作、32,376 个物理步，仿真时长 134.900 s。场景是 `tasks/task1_dimm_insert/scene/collection.usda`。完整配置与预先确定的对照容差在 `tools/task1_execution_v004/contract.json`。

物理 240 Hz、三路相机更新时序、IK 0.05 mm/0.0005 rad 精度、关节限位、400 次求解预算、动作及安全阈值保持不变。沿用基准 WARN 诊断模式；完整回放仍要求实际安全超限为零。DP 使用相同 EMA 权重 `full_flow_bc_v001/train/last.pt`、DDIM 16 步、种子 20261035，动作归一化和一次执行一步不变。

## 性能

|运行|100 ms 周期数|累计现实秒|周期中位数秒|周期 P95 秒|IK 秒|
|---|---:|---:|---:|---:|---:|
{table}

短窗口与完整窗口分别比较。启动时间不计入表中；完整回放的验证门开销另列在原始计时。历史完整成功回放原始观测边界耗时 2439.43 s。原渲染候选完整活动耗时 1077.91 s（17.97 分钟），含启动和退出的总时间 1113.68 s。紧邻的 60 步轻量基线/优化对照降低 27.62%，见 paired_benchmark.json。当前机器有其他 GPU 任务，负载在测试期间变化；相对历史完整回放的 55.81% 降幅不能全部归因于代码。历史对照不是独占 GPU 的配对基准，短窗口 27.62% 也不能代替完整回合的配对测量。

`cycle_timing.jsonl` 的 CPU 墙钟分类互斥，累计误差为零。GPU 队列独立计时，不与 CPU 总数相加。图表见 `short_timing_comparison.png`。

## IK 调用链与改动

`actor.py → native_boundary.step → physics_step → traced_apply_velocity → controller.tick → least_squares → Jacobian/residual → Lula FK`。

每个动作包含 24 个物理步；IK 按 50 ms 更新，两次目标不同，均保留。首次以当前关节为初值，后续以上一有效解为初值。模型在初始化时创建，原实现已复用模型；主要重复开销是有限差分的旋转矩阵转换、检查和残差计算。

原始日志字段 residual_calls 统计残差向量求值数；批量版仍保留 14 个扰动点，但它们在一次批量调用中计算。实际 Python 残差函数调用数另列为 residual_python_calls，由记录的 nfev 对应；不能把批量版的逻辑求值数误称为同样多的 Python 函数调用。

一次 Jacobian 有 7 个关节 × 正负两次，即 14 次残差/FK。批量版保留全部采样点，只合并旋转残差运算。同次求解中的基座变换和目标逆旋转保持固定，不跨状态缓存。{batch["jacobian_samples"]} 组雅可比逐值一致，5 个完整求解的解、nfev 和 njev 一致。离线速度不是完整回放速度。

解析 Jacobian 的数值核对通过，但完整接触前缀最大卡扣角偏差 7.87448°，未通过既定轨迹门槛。它不能作为本轮验收的执行版本。当前候选使用 `controller_batched_fd.py`。

额外优化：避免重复写入相同 drive 值；观测 NPZ 使用相同数组的无压缩格式；IPC 轮询从 50 ms 改为 2 ms。最终候选 `batched_io` 保留原默认视口、任务相机和渲染更新。主场景和公共控制器未改。

`batched_fast` 关闭默认视口的对照在物理回放上完全一致，活动耗时减少 22.0%；但腕部图像 MAE 约 1.92–1.94，超过原配置重复渲染的约 0.14–0.16，不能作为最终观测等价版本。相机回归门槛在 batched_io 测试前固定为每样本 MAE ≤0.5（8-bit 灰度单位），见 contract.json 和 camera_comparison.json。

## 原生短剖析

该短剖析使用带默认视口关闭的对照变体。61 个原生主线程帧，覆盖 10 个动作附近的 1 s 仿真。依据安装头文件 `kit/dev/include/carb/profiler/IProfileMonitor.h`，按时间区间分割，避免嵌套重复计时：

|互斥类别|秒|
|---|---:|
"""+"\n".join(f"|{k}|{v/1000:.4f}|" for k,v in native["main_thread_exclusive_ms"].items())+f"""

GPU 根事件样本平均 {native["gpu_root_sample_ms"]["mean"]:.3f} ms/帧：RTX {native["gpu_root_names_mean_ms"]["RTX Rendering"]:.3f} ms，后处理 {native["gpu_root_names_mean_ms"]["OmniGraph Post Processing"]:.3f} ms。61 帧有 183 个 RTX Render Tile、183 个 Synthetic Image Data RT；与三路任务视图一致。事件名称没有相机路径，不能据此分别归给左腕、右腕和外相机。

PhysXUpdate 包含回调与等待，不能把其包含时间当成求解计算时间。等待位于物理推进区间也不能单独证明是 GPU 求解慢；工作线程与设备等待仍需区分。详细剖析不用于最终性能对比。

## 接触、数据与五阶段验收

已完成的 `batched_io_full`：五阶段成功、零安全超限、动作差异为零、100 ms 边界误差最大 5.22e-9 s、4,050 张图像齐全。
记录到的 DIMM 最大接触穿透 0.03995 mm，未超过原 0.1 mm 判据。按钮峰值左 1.933 N、右 2.043 N；手指左 4.838 N、右 4.363 N；DIMM 环境峰值 4.095 N。接触审计依据实际报告，不声称穷举实体相交。

优化版完整验收：{("通过；活动边界耗时 %.2f 分钟，历史基准 %.2f 分钟，减少 %.2f%%；历史耗时口径下 30%% 工程目标%s；代码带来的完整配对降幅尚未测定。详见 controlled_execution_acceptance.json。" % (acc["candidate_active_wall_s"]/60,acc["baseline_active_wall_s"]/60,acc["active_reduction_fraction"]*100,"已达到" if acc["engineering_30_percent_goal_met"] else "未达到")) if acc else "尚在运行，未宣布通过或达到 30% 提速。"}

## 历史耗时

`historical_tests.csv` 按唯一回合目录汇总已有现实时间的测试，缺失分项计时保留为空。此前两轮 45 分钟 DP 测试，平均推理时间乘完成步数分别约 48.51 s、45.61 s，占现实时间约 1.80%、1.69%；该估计不含模型预热，不能将剩余时间全部称为物理求解。

## 策略停滞

成功回放开扣 48.8 s、抓稳脱架 86.0 s、准备位 104.5 s、完整完成 134.9 s。连续进展见 `policy_progress_comparison.png` 和各回合 `*_progress.csv`。

- DP v003 第 1 回合只获开扣事件。抓取阶段持续 134.4 s，基准为 37.1 s。尾段命令平移中位数 11.10 mm/s，实际 1.40 mm/s，目标领先约 5 mm；此前出现禁区接触、随后夹持丢失和张爪，继续移动不能形成有效任务进展。
- DP v003 第 2 回合获开扣和抓稳脱架事件。尾段命令 15.99 mm/s、实际 15.75 mm/s，执行能跟随；但横向误差增大 14.66 mm、姿态误差增大 88.85°，后续双指接触消失，DIMM 回落。这一段主要是任务动作方向与夹持持续性问题。
- 成功回放的插入段约 1 mm/s，仍持续缩小目标误差；小步幅本身不是停滞判据。

这里只记录进展，不改变动作或奖励。阶段超时标记以成功回放阶段耗时的 1.5 倍作为诊断候选，不能当作通用失败阈值。坐标误差以成功记录的最终 DIMM 位姿为参考；无夹持时的相对位移不解释为夹持滑移。

## 本轮 DP 记录

|记录|用途|状态|完整转移|奖励|
|---|---|---|---:|---:|
{dp_table}

关闭默认视口的诊断回合在 45 分钟内完成 1602 条转移，只完成开扣，奖励 0.05。首次超限在相对仿真时间 72.8 s：夹爪开度 0.244 mm，命令下降约 19.9 mm/s、实测下降 19.8 mm/s，没有目标限幅或控制器错误。右指接触 DIMM TopPress，右指力 30.62 N、DIMM 环境力 30.79 N；成功回放相近下降高度的开度约 9.8 mm。证据见 dp_episode_002/first_grasp_violation.json。该诊断不能证明保留原图像条件下的策略能力变化。

## 复现与回退

所有命令在 isaac-b 的 `/mnt/data2/wjy_SERL/Embodied_Safety` 下执行：

```bash
python3 -B tools/task1_execution_v004/run.py --variant baseline --steps 60 --name baseline_new
python3 -B tools/task1_execution_v004/run.py --variant batched_io --steps 60 --name batch_new
TASK1_WINDOW_GATES=1 python3 -B tools/task1_execution_v004/run.py --variant batched_io --steps 1349 --name full_new
```

每次使用新的产物名称。回退只需选择 `--variant baseline`，原生产入口仍保留。

## 已知限制

- 完整优化回放和 DP 测试以各自验收 JSON 为准；短窗口提速不能代替完整回放通过。
- 原生 GPU 事件为异步样本，不能严格对应每一个观测周期，也未映射到单路相机。
- 图片/日志仍同步写入；当前主要瓶颈是 IK 与原生等待，未启用异步写队列。视频在回合后生成。
- 撤下的短 DP 诊断记录在 `dp_episode_001/manual_interruption.json`，不计作策略成功或失败。
- 新增代码只在 `tools/task1_execution_v004/`；主场景、权重、训练代码和 baselines 未改。
"""
final_path=R/"dp_episode_003/status.json"
if final_path.exists():
    final=read(final_path)
    if final.get("status")=="finished":
        result=final.get("result",{})
        progress=read(R/"dp_episode_003_progress.json")
        timing=read(R/"dp_episode_003/eval/episode_01/timing_summary.json")
        text += "\n## 原渲染单轮策略验收\n\n"
        text += f"现实时间 {final['wall_s']/60:.2f} 分钟，仿真时间 {final['simulation_elapsed_s']:.2f} s，完成 {final['full_transitions']} 条转移，奖励 {final.get('reward')}。完整任务成功：{result.get('full_task_success',False)}；结束原因：{result.get('failure_reason')}。\n\n"
        text += f"平均纯推理 {final.get('inference_mean_ms'):.3f} ms；预处理加推理 {final.get('preprocess_plus_inference_mean_ms'):.3f} ms。周期中位数 {timing['cycle_median_s']:.3f} s，P95 {timing['cycle_p95_s']:.3f} s。\n\n"
        text += "|本轮策略执行分项|秒|活动周期占比|\n|---|---:|---:|\n"
        for key in ("controller_tick","kit_native_unresolved","contact_report","ipc_poll_wait","image_encode_write","physics_log_serialize_write"):
            value=timing["modules"][key]["total_s"]
            text += f"|{key}|{value:.2f}|{100*value/timing['active_wall_s']:.2f}%|\n"
        text += "\n策略计算发生在独立进程，部分包含在 actor 的 IPC 等待中，不能再与上述分项重复相加。\n\n"
        text += "阶段事件："+json.dumps(progress['events'],ensure_ascii=False)+"。\n\n"
        text += "末段 30 s 进展："+json.dumps(progress["tail_windows"][-1],ensure_ascii=False)+"。\n\n"
        text += "本轮末段命令与实际速度中位数均约 8.21 mm/s，目标距离缩短约 151 mm，不能将阶段未通过解释为完全没有几何进展。末段仍有 forbidden_contact，许可区抓取和姿态等条件未通过；已有阶段奖励不会因后续失败自动撤销。失败回合保持失败标签。\n\n"
        text += "连续进展与接触诊断见 dp_episode_003_progress.json、dp_episode_003_progress.csv 和 stagnation_evidence.json；这些记录与固定动作回放的执行验收分别报告。视频：dp_episode_003/replay/actual_policy_replay.mp4。\n"
contact_path=R/"dp_episode_003/first_grasp_violation.json"
if contact_path.exists():
    c=read(contact_path)
    text += "\n## 原渲染策略的抓取接触证据\n\n"
    text += f"首次抓取力超限出现在相对仿真时间 {c['relative_time_s']:.1f} s（动作 {c['index']}）。夹爪开度 {c['gripper_width_mm']:.3f} mm，命令 Z 速度 {c['command_translation_mm_s'][2]:.2f} mm/s，实际 Z 速度 {c['actual_translation_mm_s'][2]:.2f} mm/s。该周期目标限幅 {c['goal_lead_limited_steps']} 次，控制器错误 {c['controller_error']}。\n\n"
    text += f"右指峰值 {c['force_peaks_N']['finger_right_N']:.2f} N，DIMM 环境峰值 {c['force_peaks_N']['dimm_environment_N']:.2f} N。成功回放相近下降高度的夹爪开度为 {c['matched_success_descent']['gripper_width_mm']:.2f} mm；对应位置差 XYZ 为 {c['matched_success_descent']['position_difference_mm']} mm。该对照按下降高度匹配，不是完整位姿相等。\n\n"
    text += "此处的直接问题是夹爪接近闭合后仍持续下压，并发生实际阻挡接触；没有 IK 失败证据。精确动作、前后观测与该 100 ms 内接触点保存在 dp_episode_003/first_grasp_violation.json。该证据不单独解释模型为何提前闭爪。\n"
(R/"REPORT.md").write_text(text)
(R/"report_summary.json").write_text(json.dumps(dict(workflow=status,timing=records,optimized_full_acceptance=acc),indent=2))
patch=[];stats=[]
for p in sorted(D.glob("*")):
    if p.is_file():
        x=subprocess.run(["git","diff","--no-index","--","/dev/null",str(p)],cwd=ROOT,capture_output=True,text=True)
        patch.append(x.stdout)
        stats.append(subprocess.run(["git","diff","--no-index","--stat","--","/dev/null",str(p)],cwd=ROOT,capture_output=True,text=True).stdout)
(R/"code_changes.patch").write_text("".join(patch))
(R/"code_changes_stat.txt").write_text("".join(stats))
(R/"new_files.txt").write_text("\n".join(str(p) for p in sorted(D.iterdir()) if p.is_file())+"\n")
print("Wrote REPORT.md, timing_comparison.csv, 2 figures, code_changes.patch")
