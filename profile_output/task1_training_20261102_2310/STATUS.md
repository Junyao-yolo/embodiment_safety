# Task1 训练保存与性能测试状态

正式训练暂停。GPU 独占测试等待 Task2 释放 RTX 4090；不暂停 Task2，不使用 CPU 训练，不自动恢复正式训练。尚无本次独占计时或加速收益结果。

服务器：`isaac-b`（ufdwaa）。项目：`/mnt/data2/wjy_SERL/Embodied_Safety`。

保存目录：`/mnt/data2/wjy_SERL/Embodied_Safety/profile_output/task1_training_20261102_2310`。

| 检查点 | 更新数 | batch | 采样游标 |
|---|---:|---:|---:|
| `batch4_step5000.pt` | 5000 | 4 | 26000 |
| `batch16_step500.pt` | 500 | 16 | 8000 |

两份检查点均包含模型、EMA、AdamW、步数、配置、采样排列及游标、Python/NumPy/Torch/CUDA随机状态。batch=4分支前500步使用batch=16，之后4500步使用batch=4，不等价于全程batch=16。

本次检查时训练进程已退出。batch=4最后日志为7626步，但最新完整检查点为5000步；5001—7626步未落盘，无法补存。原始归档与日志保留，完整检查点另存副本。

最后实际训练入口为`algos.diffusion_policy.train`，命令行batch=4、预算100000步、三路相机×两帧、256×256 RGB。启动脚本指定`CUDA_VISIBLE_DEVICES=0`。当前磁盘实验配置已变成batch=16，当前没有正式训练进程，不能据此把之前的实际batch说成16。退出进程的完整环境变量无法追溯。

数据路径：`/mnt/data2/wjy_SERL/Embodied_Safety/outputs/datasets/task1/diffusion_success100_v001`。100条轨迹，90条训练、10条验证，124592个训练窗口，13853个验证窗口。正式预算未改动。

最后batch=4训练没有启用压缩图像缓存。SequenceData直接读取机械硬盘`/dev/sda3`上的未压缩`.npy`，不解码PNG/JPEG。三个图像数组合计82.54 GB，训练实际覆盖的图像约73.54 GB。旧训练1334窗口覆盖约0.79 GB图像。历史日志复算：旧版43.624 ms/更新；batch=4最后500行对应499个间隔，159.383 ms/更新。这些历史数字不是独占GPU对照。

完整无损压缩缓存已经复制到NVMe：`/home/ufdwaa/.cache/task1_training_20261102_2310/rgb`，共20,640,146,309字节，复制并落盘耗时95.44秒。全部图像与索引保留。该缓存使用zlib和横向差分重建；属于无损存储副本，没有应用层LRU。操作系统页缓存是否命中需结合实际磁盘读取量分析。

测试队列为`queue_profile.py`，每组200次预热＋500次稳定更新：

1. 旧完整配置：batch=8，原数据与原scope。
2. 当前完整数据：batch=4，机械硬盘原始数组。
3. 相同batch=4检查点与采样：机械硬盘无损压缩缓存。
4. 相同batch=4检查点与采样：NVMe完整无损压缩缓存。
5. 相同batch=4检查点与采样：NVMe缓存，uint8传输到GPU后归一化。

前向、反向、梯度裁剪、优化器和EMA均在GPU。所有组采用同样的主线程计时与CUDA事件，保留原版的有限性检查、loss回读与日志刷新行为；记录数据等待、H2D、前向、反向、优化器、EMA、日志及同步开销。验证和完整检查点保存含fsync单独计时。记录输入差异、采样索引、每步loss和验证指标。

逐秒GPU遥测包括利用率、显存、功耗、时钟和进程，同时记录磁盘统计、内存与换页。其他计算任务出现时停止并剔除该组测试，等待下次空闲窗口。桌面图形进程保留并记录。每组只对本组图像文件请求清理页缓存，不清空全机缓存。

实时状态见`queue_status.json`；完整结果生成后见`REPORT.md`、`analysis.json`、各组`update_timing.jsonl`与`*_gpu.jsonl`。短测检查点命名`benchmark_only.pt`，不并入正式训练进度。

| 文件 | 变化 | 内容 |
|---|---|---|
| `profile_train.py` | 新增 | 独立GPU计时、分项数据读取记录、完整验证与保存 |
| `queue_profile.py` | 新增 | 空闲GPU等待、资源监控、完整缓存副本和测试队列 |
| `report_profile.py` | 新增 | 速度、资源与短测数值差异汇总 |
| `preservation.json`、`configuration_audit.json`、`working_sets.json` | 新增 | 检查点与数据配置证据 |
| 正式训练源码及配置 | 未修改 | 等报告后决定恢复方案 |
