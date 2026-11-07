# 复现入口

先区分三个目标。公开代码可以检查通用执行、经验表示、配对统计和合成故障路径。原服务器保留的私有资产可以只读复核真实运行。迁移到新环境后重新执行模型实验，需要取得外部资产并登记新研究。

## 公开 CPU 检查

Python 3.11 或更新版本，在仓库根目录执行。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
python -m wll.cli --help
CUDA_VISIBLE_DEVICES= PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
```

公开交付副本的全部 596 项 CPU 测试通过。合成测试覆盖样本隔离、工具拒绝证据、候选筛查、配对控制循环、评分失败和释放失败等程序行为。它们不调用真实模型，不能作为模型成功率。测试只保留公开核心、当前闭环及其实际依赖，其他历史研究的测试仍在原工作区保留。

## 真实闭环的阅读顺序

| 环节 | 入口 | 核验目标 |
| --- | --- | --- |
| 受控工具纠错 | `scripts/run_native_state_recovery_fixture_001.py` | 前读、未执行的拒绝、同参纠正、同资源后读 |
| 顺序证据 | `scripts/native_state_recovery_fixture_transitions_001.py` | 仅投影允许的匿名事实，不加入题目答案 |
| 经验生成与复审 | `scripts/run_native_state_recovery_transition_writer_002.py`、`scripts/native_state_recovery_transition_review_002.py` | 一次生成、结构筛查、原文复审、候选冻结 |
| 自编面板 | `scripts/run_native_state_recovery_authored_study_003.py` | 两臂新环境、配对执行、独立保存状态评分 |
| 官方面板 | `scripts/freeze_native_state_recovery_official_003.py`、`scripts/run_native_state_recovery_official_003.py` | 排除已使用场景、冻结顺序与预算、官方评分 |
| 只读审计 | `scripts/report_native_state_recovery_official_001.py`，由 003 入口绑定 | 从关闭回执、评分日志和资源释放记录重算终态 |

最新聚合证据在 `evidence/state-recovery-official-003.json`。源文件中的版本与哈希用于保留历史执行绑定，跨环境使用需要重新登记，不能直接修改旧绑定后沿用旧成绩。

公开版的精确 token 计数沿用本地 CPU 子进程入口，通过 `WLL_TOKENIZER_PATH` 指定冻结分词器目录。原生研究模块需要的解释器及 SSH 配置名仍保留，旧批处理和固定服务器预检继续受各自的环境绑定约束。

## 原环境的只读复核

下面命令仅适用于保留完整私有登记、模型及运行回执的原远端工作区。公开仓库没有这些资产，因此不能单独运行该命令。它检查已有证据，不会重新执行任务或评分。

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python -B - <<'PY'
import json
from pathlib import Path
from scripts import run_native_state_recovery_official_003 as study

saved = json.loads(Path(study.REPORT).read_text())
assert study.report(saved['plan_sha256'], saved['result_sha256']) == saved
print(saved['utility'])
PY
```

## 在新环境执行

需要自行取得 AppWorld 数据及对应版本、模型与 tokenizer、B3 adapter、训练与推理依赖、隔离 worker 和官方评分环境。部署脚本保留原服务器目录、systemd 服务及设备策略，不能承诺安装 Python 包后即复现完整 GPU 实验。

迁移时先核对模型和数据版本，再配置私有目录与隔离执行器。使用合成输入检查执行、保存、评分和关闭，然后按来源与评测场景隔离原则生成新的登记。GPU 入口继续遵守设备白名单、180 秒连续空闲观察、单次票据、项目锁和 20 GiB 分配上限。未验证独立队列前使用串行调度。

旧盲测面板已经使用，不能用于调提示或补跑。新的模型、预算或场景研究需要独立结果目录与预登记。此仓库公开实现、合成输入和聚合结果，不公开受保护任务、交互轨迹、经验原文或权重。完整新环境的真实模型复现尚未验证。
