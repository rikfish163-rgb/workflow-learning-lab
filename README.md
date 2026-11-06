# Workflow Learning Lab

基于 AppWorld 的 LLM Agent 训练、工作流纠错与评测项目。

项目研究如何从执行轨迹中识别可靠的恢复行为，再检验提取的经验能否帮助其他任务。当前已完成动作级微调、来源任务评测和两例受控漏写恢复实验。经验提取的实际调用返回弃权，跨任务收益仍待验证。

## 已完成的工作

- 使用 NF4 量化与 LoRA，在单张 24GB GPU 上完成 27B 模型成功轨迹 SFT，共 344 条动作样本、86 次参数更新。训练保留完整历史，仅监督目标助手动作。
- 实现 CodeAct 沙箱执行与原生 JSON 工具两条执行路径，记录生成、工具调用、预算停止和任务终态。
- 完成 12 项 CodeAct 来源任务评测，其中 8 项官方通过。任务属于已暴露的训练场景。
- 在两个受控遗漏场景中核验模型补回缺失写入的行为，检查目标、接口、参数和返回的一致性。
- 实现经验提取、候选审核及完整配对评测模块。真实提取调用包含弃权结果，目前没有通过审核的经验进入 27 对任务比较。

详细口径见 [实验结果](docs/RESULTS.md)，讲解思路见 [面试准备](docs/INTERVIEW.md)。

## 代码入口

| 环节 | 文件 |
| --- | --- |
| 通用运行与比较命令 | [wll/cli.py](wll/cli.py) |
| 任务、轨迹与配对统计 | [wll/protocol.py](wll/protocol.py) |
| 执行循环与恢复 | [wll/runner.py](wll/runner.py)、[wll/recovery.py](wll/recovery.py) |
| 训练样本与目标损失 | [wll/training.py](wll/training.py)、[scripts/qwen38_training.py](scripts/qwen38_training.py) |
| 显存受限的更新实现 | [scripts/hybrid_target_updates.py](scripts/hybrid_target_updates.py) |
| CodeAct 工作流执行 | [scripts/codeact_within_episode_001.py](scripts/codeact_within_episode_001.py) |
| 工具执行与隔离 | [scripts/owned_appworld_v3.py](scripts/owned_appworld_v3.py)、[scripts/native_appworld_replay.py](scripts/native_appworld_replay.py) |
| 经验生成与审核 | [scripts/native_experience_writer_001.py](scripts/native_experience_writer_001.py)、[scripts/native_experience_content_001.py](scripts/native_experience_content_001.py) |
| 配对统计与场景级重采样 | [scripts/native_experience_utility_001.py](scripts/native_experience_utility_001.py)、[scripts/exact_cluster_resampling.py](scripts/exact_cluster_resampling.py) |

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

上述测试使用合成输入验证程序行为。运行真实 AppWorld 任务需要另行准备数据、模型、容器环境和实验登记。训练组件需要对应的 PyTorch、Transformers 和 PEFT 环境。调用固定模型的精确 token 计数时，通过 `WLL_TOKENIZER_PATH` 指定本地分词器目录。

## 评测方法

模型自报完成和官方终态评分分别记录。恢复后的成功带有额外执行机会，和初试结果分别统计。跨任务评测设计固定 9 个场景、27 对任务、54 次新环境执行，两臂使用同一模型、随机种子和预算，仅改变经验输入。场景级统计处理同场景多题的相关性，缺失或未可靠关闭的结果保留为 unknown。

## 数据与发布范围

公开版包含核心实现、合成测试、实验结果和面试说明。真实任务的输入、交互轨迹、评分回执、模型权重和原环境的调度配置独立管理。实验数字采用 [实验结果](docs/RESULTS.md) 标注日期的已核验快照。
