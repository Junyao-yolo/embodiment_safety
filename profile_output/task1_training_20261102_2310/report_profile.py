"""Summarize isolated timing, resource use, and matched numerical comparisons."""
import json
from pathlib import Path
import statistics
import numpy as np

ROOT = Path('/mnt/data2/wjy_SERL/Embodied_Safety')
OUT = ROOT/'profile_output/task1_training_20261102_2310'


def history(path, old=False):
    rows = [json.loads(s) for s in path.read_text().splitlines()]
    if not old:
        rows = rows[-500:]
    intervals = [b['elapsed_s']-a['elapsed_s'] for a, b in zip(rows, rows[1:])
                 if 'validation' not in a and 'validation' not in b
                 and b['elapsed_s'] >= a['elapsed_s'] and (not old or a['step'] >= 1000)]
    return dict(mean_ms=statistics.mean(intervals)*1000,
                median_ms=statistics.median(intervals)*1000, intervals=len(intervals))


def main():
    results = json.loads((OUT/'results.json').read_text())
    original = ROOT/'outputs/runs/task1/diffusion_policy/full_flow_bc_v001/train/metrics.jsonl'
    prior = dict(old=history(original, True), current=history(OUT/'batch4_archived_metrics.jsonl'))
    base = next(r for r in results if r['name']=='current_batch4_raw')
    basepath = Path(base['output'])
    baseids = np.load(basepath/'sample_indices.npy')
    baseloss = np.array([json.loads(s)['loss'] for s in (basepath/'metrics.jsonl').read_text().splitlines()])
    for row in results:
        path = Path(row['output'])
        samples = [json.loads(s) for s in Path(row['telemetry']).read_text().splitlines()]
        measured = [s for s in samples if row['stable_timestamp_start'] <= s['timestamp'] <= row['stable_timestamp_end']]
        stats = {}
        for key in ('utilization_pct', 'memory_mib', 'power_w', 'sm_clock_mhz'):
            values = [s[key] for s in measured if s.get(key) is not None]
            stats[key] = dict(mean=statistics.mean(values), max=max(values), min=min(values)) if values else None
        row['gpu_stable'] = stats
        row['all_observed_processes'] = {p['pid']: p for s in samples for p in s['processes']}
        row['other_active_compute_seen'] = any(int(p['pid']) != s['trial_pid'] for s in samples for p in s['active_compute'])
        if row['name'].startswith('current'):
            ids = np.load(path/'sample_indices.npy')
            losses = np.array([json.loads(s)['loss'] for s in (path/'metrics.jsonl').read_text().splitlines()])
            row['comparison'] = dict(sample_indices_equal=bool(np.array_equal(baseids, ids)),
                  finite_losses=bool(np.isfinite(losses).all()),
                  loss_max_abs_delta=float(np.max(np.abs(losses-baseloss))),
                  loss_mean_delta=float(losses.mean()-baseloss.mean()),
                  validation_mse_relative=row['validation']['action_mse']/base['validation']['action_mse'],
                  speedup=base['update_mean_ms']/row['update_mean_ms'])
    (OUT/'analysis.json').write_text(json.dumps(dict(history=prior, trials=results), indent=2)+'\n')
    lines = ['# Task1 DP 训练性能实测', '',
        '正式训练保持暂停；性能短测不计入正式更新预算。batch=4 第5000步与 batch=16 第500步完整检查点分别保留。batch=4 分支的前500次更新使用 batch=16，之后使用4，二者不等价。', '',
        '所有组使用200次预热、500次稳定更新；每组从各自检查点独立恢复。新数据组使用相同第5000步检查点、采样顺序、随机状态和完整124592个训练窗口。旧数据组保留原batch=8与1334窗口设置，用于复核历史速度。', '',
        '每组开始前仅对该组图像文件请求POSIX_FADV_DONTNEED；未清空全机页缓存。稳定段包含200次预热形成的自然页缓存。缓存和NVMe对照均保留全部139945帧索引×三路图像。', '',
        '| 组别 | batch | ms/更新 | 窗口/s | GPU利用率均值 | 显存峰值 MiB | 功耗均值 W | 验证 s | 保存含fsync s |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in results:
        g = r['gpu_stable']
        def val(key, stat='mean'):
            return f'{g[key][stat]:.1f}' if g[key] else '未采到'
        lines.append(f"| {r['name']} | {r['batch_size']} | {r['update_mean_ms']:.2f} | {r['windows_per_second']:.2f} | {val('utilization_pct')} | {val('memory_mib','max')} | {val('power_w')} | {r['validation_s']:.2f} | {r['checkpoint_save_fsync_s']:.2f} |")
    lines += ['', '主线程分项均值（毫秒/更新；CUDA分项另见analysis.json，不与主线程耗时相加）：', '',
              '| 组别 | 数据等待 | H2D提交 | 前向提交 | 有限性同步 | 反向提交 | 梯度裁剪 | 优化器提交 | EMA提交 | loss回读 | 日志IO |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    keys = ('data_wait','h2d','forward','finite_check','backward','gradient_clip','optimizer','ema','loss_to_cpu','log_io')
    for r in results:
        lines.append('| '+r['name']+' | '+' | '.join(f"{r['regions'][k+'_host_ms']['mean']:.3f}" for k in keys)+' |')
    lines += ['', 'CUDA事件分项均值（毫秒/更新）：', '',
              '| 组别 | H2D | 前向 | 反向 | 梯度裁剪 | 优化器 | EMA |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for r in results:
        lines.append('| '+r['name']+' | '+' | '.join(f"{r['regions'][k+'_cuda_ms']['mean']:.3f}" for k in ('h2d','forward','backward','gradient_clip','optimizer','ema'))+' |')
    lines += ['', '历史日志按相邻更新elapsed差值、排除验证保存边界复算：', '',
              f"- 旧版：{prior['old']['mean_ms']:.3f} ms/更新，{prior['old']['intervals']}个间隔。",
              f"- batch=4末段：{prior['current']['mean_ms']:.3f} ms/更新，{prior['current']['intervals']}个间隔。", '',
              '修复收益与相同700次更新的数值核对：', '',
              '| 对照 | 加速倍数 | 索引一致 | loss最大绝对差 | 验证MSE/基线 |',
              '|---|---:|---|---:|---:|']
    for r in results:
        if 'comparison' in r:
            c = r['comparison']
            lines.append(f"| {r['name']} | {c['speedup']:.3f} | {c['sample_indices_equal']} | {c['loss_max_abs_delta']:.8g} | {c['validation_mse_relative']:.5f} |")
    hot = max(((k, base['regions'][k+'_host_ms']['mean']) for k in keys), key=lambda x:x[1])
    lines += ['', f'当前原始读取基线的最大主线程分项为 {hot[0]}，均值 {hot[1]:.3f} ms/更新。', '',
              '原始SequenceData读取的是未压缩.npy，不解码JPEG/PNG。压缩缓存每次请求执行zlib解压和横向差分重建；它是完整的无损存储缓存，没有应用层LRU，也不能把所有请求称为内存命中。实际解码次数、读取字节数、major fault与内核磁盘读取量见analysis.json。', '',
              '## 已知限制', '',
              '历史进程已经退出，无法追溯其完整环境与当时GPU争抢程度。新计时逐秒记录GPU进程、功耗和利用率；若有其他计算进程出现，整组停止并剔除，等待下个空闲窗口重测。桌面图形进程保留并记录。分项CUDA事件会有少量观测开销，所有组使用相同方法。短测数值一致性不代表完成10万步后的任务成功率。', '',
              '## 性能测量文件', '',
              '| 文件 | 类型 | 内容 |', '|---|---|---|',
              '| profile_train.py | 新增 | 200+500更新、分项计时、完整验证与短测检查点 |',
              '| queue_profile.py | 新增 | 等待空闲GPU、进程遥测、完整缓存NVMe副本、独立短测队列 |',
              '| report_profile.py | 新增 | 同口径汇总及数值比较 |',
              '| 各组update_timing.jsonl / *_gpu.jsonl | 新增 | 每次更新及逐秒资源记录 |',
              '| 正式训练源码与配置 | 未修改 | 等待恢复方案决定 |', '']
    (OUT/'REPORT.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
