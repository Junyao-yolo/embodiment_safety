# Task1 wiring

当前目录图见 [README.md](README.md) 和 [LAYOUT.md](LAYOUT.md)。Task1 说明：[Task1](tasks/task1_dimm_insert/README.md)。下文是 SERL 入口。

`runners.prepare_task1` → 当前示范索引与在线 replay
`runners.preflight` → 代码、资产依赖及数据哈希检查
`runners.train` → 固定数据上的示范/在线混合 DrQ-SAC 更新
`runners.rollout --learn` → 原生 Isaac 回合与在线混合更新
`runners.eval` → 确定性策略回合，独立评估 replay，不更新权重

回合重置使用新进程重新加载场景，不通过修改关节角度模拟任务完成。
