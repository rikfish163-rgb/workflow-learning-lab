# Workflow Learning Lab

基于 AppWorld 的 LLM Agent 训练、工作流纠错与评测项目。

项目研究如何从执行轨迹中识别可靠的恢复行为，再检验提取的经验能否帮助其他任务。已完成动作级微调、四个受控工具纠错工作流、模型经验提取和独立跨任务评测。自编面板两臂各通过相同的 3/27。独立官方 `test_normal` 面板完成 9 场景、27 对任务和 54 份评分，两臂均为 0/27，未观察到成功率提升。

官方 003 经授权从未使用场景另行冻结，与旧中断面板没有场景重叠。两臂共同使用登记的 75 秒单次生成限制，54 份评分及 14 次 GPU 释放均已核验。旧运行保留原终态，未重跑或回填。 [003 完整终态](evidence/state-recovery-official-003.json) · [003 执行入口](scripts/run_native_state_recovery_official_003.py)

## 已完成的工作

- 使用 NF4 量化与 LoRA，在单张 24GB GPU 上完成 27B 模型成功轨迹 SFT，共 344 条动作样本、86 次参数更新。训练保留完整历史，仅监督目标助手动作。
- 实现 CodeAct 沙箱执行与原生 JSON 工具两条执行路径，记录生成、工具调用、预算停止和任务终态。
- 完成 12 项 CodeAct 来源任务评测，其中 8 项官方通过。任务属于已暴露的训练场景。
- 在两个受控遗漏场景中核验模型补回缺失写入的行为，检查目标、接口、参数和返回的一致性。
- 在四个受控工作流中核验同一资源的前读、写入前拒绝、一次同参纠正和后读。
- 从匿名顺序证据中生成一条通过结构筛查及模型复审的候选经验。自编面板完成 27 对、54 份独立评分和 14 次 GPU 释放审计。

详细口径见 [实验结果](docs/RESULTS.md)，讲解思路见 [面试准备](docs/INTERVIEW.md)，环境与复现范围见 [复现入口](docs/REPRODUCING.md)。

## 代码入口

| 环节 | 文件 |
| --- | --- |
| 通用运行与比较命令 | [wll/cli.py](wll/cli.py) |
| 任务、轨迹与配对统计 | [wll/protocol.py](wll/protocol.py) |
| 执行循环与恢复 | [wll/runner.py](wll/runner.py)、[wll/recovery.py](wll/recovery.py) |
| 训练样本与目标损失 | [wll/training.py](wll/training.py)、[scripts/qwen38_training.py](scripts/qwen38_training.py) |
| 显存受限的更新实现 | [scripts/hybrid_target_updates.py](scripts/hybrid_target_updates.py) |
| CodeAct 工作流执行 | [scripts/codeact_within_episode_001.py](scripts/codeact_within_episode_001.py) |
| 工具执行与隔离 | [scripts/owned_appworld_v3.py](scripts/owned_appworld_v3.py)、[scripts/native_appworld_controller_006.py](scripts/native_appworld_controller_006.py) |
| 经验生成与审核 | [scripts/native_experience_writer_001.py](scripts/native_experience_writer_001.py)、[scripts/native_experience_content_001.py](scripts/native_experience_content_001.py) |
| 完整配对评测 | [scripts/native_experience_utility_001.py](scripts/native_experience_utility_001.py) |
| 受控纠错与顺序证据 | [scripts/run_native_state_recovery_fixture_001.py](scripts/run_native_state_recovery_fixture_001.py)、[scripts/native_state_recovery_fixture_transitions_001.py](scripts/native_state_recovery_fixture_transitions_001.py) |
| 已审核经验与最终评测 | [scripts/native_state_recovery_transition_review_002.py](scripts/native_state_recovery_transition_review_002.py)、[scripts/run_native_state_recovery_official_003.py](scripts/run_native_state_recovery_official_003.py) |
| 结果公开投影 | [scripts/report_paired_omission_utility_001.py](scripts/report_paired_omission_utility_001.py) |

脚本文件名中的版本后缀用于保留各次实验的执行代码。阅读时可先从 wll 核心和上表入口开始。

## 本地 CPU 检查

需要 Python 3.11 或更新版本。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
python -m wll.cli --help
CUDA_VISIBLE_DEVICES= PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
```

公开交付副本的全部 596 项 CPU 测试通过。上述测试使用合成输入验证程序行为。运行真实 AppWorld 任务还需要自行取得相应数据、模型、容器环境和实验登记输入。scripts 中的历史训练及 GPU 控制器包含原服务器路径和哈希绑定，迁移环境后需要重新登记；完整真实实验并非安装 Python 包后即可直接复现。

## 评测方法

模型自报完成和官方终态评分分别记录。恢复后的成功带有额外执行机会，和初试结果分别统计。跨任务评测设计固定 9 个场景、27 对任务、54 次新环境执行，两臂使用同一模型、随机种子和预算，仅改变经验输入。场景级统计处理同场景多题的相关性，缺失或未可靠关闭的结果保留为 unknown。

## 数据与发布范围

此仓库提供实现、合成测试和汇总说明。AppWorld 受保护任务内容、原始交互轨迹、凭据及模型权重另行管理。完整实验依赖的私有登记输入和模型资产未打包。

最新研究状态见 [实验结果](docs/RESULTS.md)。自编与官方研究的聚合终态保存在 [evidence](evidence)。
