# Documentation

[Project overview](../README.md) · [中文首页](../README.zh-CN.md) · [Roadmap](../ROADMAP.md)

OpenJev has research specifications and tools for conversation-data preparation, offline review, and reference scoring; no OpenJev model has been released. Detailed technical specifications are written in Chinese; the project overview and contribution documents are available for an international audience.

## Suggested reading order / 建议阅读顺序

| Order | Document | What it explains |
| --- | --- | --- |
| 1 | [总体设计](openjev_design.md) | 技术选择、官方公开行为与 OpenJev 假设、实施顺序 |
| 2 | [模型架构详解](openjev_model_architecture.md) | Backbone、输入编译、A／B 候选结构、attention、读出与输出 |
| 3 | [训练流程与数据合成](openjev_training_pipeline.md) | 每一步的数据来源、合成方法、训练目标、评估和退出条件 |
| 4 | [Roadmap](../ROADMAP.md) | 首个模型预览版发布承诺和长期研究里程碑 |
| 5 | [Model card](MODEL_CARD.md) | 当前未发布状态与发布时需要提供的模型信息 |
| 6 | [Release guide](RELEASING.md) | 可复现模型发布的产物、验证和元数据要求 |

## Active data work / 当前数据工作

当前数据路线：[从现有聊天语料构造决策题](openjev_training_pipeline.md)。先看[本地语料盘点](data_audit/README.md)，了解可用来源、重复与来源信息缺口，再看制题、本地概率采集和验证流程。

当前保留[五份冻结来源](data_audit/README.md)，共 6,832 个用户轮次前缀，以及本地教师的 255 编码审计。旧硬标签批次、远程预检及对应脚本、配置、报告已[归档](ARCHIVE.md)，不再作为当前入口。

多语言、候选长度及 K=2–255 的覆盖要求见[多样性与 PiSSA 方案](openjev_training_pipeline.md)。先修复当前制题缺陷，再进行 1K–3K 的 B＋PiSSA 闭环；命令见[执行说明](execution.md)。

训练路线统一为准备数据 → PiSSA 直接软蒸馏 → 验证。API 从对话构造问题与选项，本地教师 thinking 后提供完整候选分布，不做硬标签训练。历史远程预检已归档，[255 编码审计](data_audit/teacher_codebook.md)已完成；无标签制题、审核、回放与本地概率采集入口已实现；实际质量和覆盖见[小试报告](data_audit/pipeline_canary.md)。软训练器仍待实现。外部 state／questions／criteria 及响应字段见 [Jev 接口约定](openjev_api_contract.md)，包括驾驶及三类型示例、confidence 本地定义与实现边界。[数据格式](openjev_training_data.md)说明公共 input、无标签题库、软监督与冻结约束。

旧的合成规则 Step 0 实验已移出当前代码与文档入口，存放位置及恢复说明见[归档记录](ARCHIVE.md)。

## Contributing and repository information

- [Contributing](../CONTRIBUTING.md)
- [Code of Conduct](../CODE_OF_CONDUCT.md)
- [Security](../SECURITY.md)
- [Changelog](../CHANGELOG.md)
- [License](../LICENSE) and [third-party notices](../THIRD_PARTY_NOTICES.md)
- [Citation metadata](../CITATION.cff)

## Research references

[Jev 官方资料入口](references.md)集中记录接口文档与发布文章的来源。原网页和文章摘录已移入本地归档；它们不是 OpenJev 的实现规格或本项目实测结果。第三方权利说明见[third-party notices](../THIRD_PARTY_NOTICES.md)。
