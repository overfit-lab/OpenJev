<p align="center">
  <img src="docs/assets/openjev-banner.svg" alt="OpenJev — 面向软件的类型化决策模型" width="100%" />
</p>

<p align="center">
  <strong>面向软件的开源决策模型。</strong><br />
  直接输出可供代码使用的决策与概率，服务自动化工作流。
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-74e0c1" alt="许可证：Apache-2.0" /></a>
  <a href="ROADMAP.md"><img src="https://img.shields.io/badge/status-design%20stage-7aa2f7" alt="当前阶段：设计" /></a>
  <a href="ROADMAP.md"><img src="https://img.shields.io/badge/model%20preview-coming%20soon-f5c97a" alt="模型预览版即将发布" /></a>
</p>

<p align="center">
  <a href="README.md">English</a> · 简体中文 · <a href="docs/README.md">文档</a> · <a href="ROADMAP.md">路线图</a> · <a href="CONTRIBUTING.md">参与贡献</a>
</p>

> **首个模型预览版将在几天内发布。**
> 预览版将提供模型权重、推理示例和初步评估，进度见[路线图](ROADMAP.md)。

目前仓库包含架构设计和训练方案，代码与模型权重将随预览版发布。

## 为什么做 OpenJev？

工单路由、政策检查、报告评分，这些判断应该能像普通函数一样被软件调用。OpenJev 的设计是读取状态、问题和候选，直接返回决策与概率分布。

业务代码负责设置阈值、处理不确定的情况。模型直接为候选打分，省去答案字符串的生成过程。

项目围绕四件事展开：

- **类型明确：**候选由调用方定义，输入验证和确定性后处理保证成功响应的值合法。
- **概率质量：**使用可靠标签和概率监督训练，在独立数据上校准。
- **推理高效：**省去词表解码，研究多个决策之间复用状态和问题计算。
- **过程可复现：**发布模型产物、配置、评估流程及各版本的能力边界。

OpenJev 受 Jev 公开资料启发，与 TypeSafe AI、Qwen 没有隶属关系。下文介绍我们的架构和训练方法。

## 模型会输出什么？

三个计划支持的 primitive 复用一个评分模型：

| 类型 | 问题形式 | 预期输出 |
| --- | --- | --- |
| **Choice** | 哪个候选最符合当前输入？ | 选中的候选 ID 和完整候选概率 |
| **Score** | 当前输入落在哪些描述性档位？ | 档位概率、加权分数和分布方差 |
| **Noul** | 这个命题是否成立？ | 回答为真的概率 |

以工单为例：

```text
状态：快递显示签收，但客户说没有收到。
问题：首先应该交给哪个团队？
候选：shipping / billing / returns

模型 → 候选分数 → 概率分布 → 业务路由代码
```

拟定的输出格式如下，概率为示例值：

```json
{
  "type": "choice",
  "selected_id": "shipping",
  "probabilities": {
    "shipping": 0.90,
    "billing": 0.03,
    "returns": 0.07
  },
  "top_probability": 0.90,
  "margin": 0.83
}
```

输出验证负责保证取值合法，独立评测负责衡量判断质量和工作流风险。

## 技术路线

```mermaid
flowchart LR
    A[状态与类型化问题] --> B[输入编译器]
    B --> C[Qwen3-1.7B-Base]
    C --> D[共享标量评分头]
    D --> E[按问题归一化的概率]
    E --> F[类型化输出]
    F --> G[工作流决策策略]
```

### 候选评分

保留 Qwen3-1.7B-Base 的 embedding 和 Transformer 层，增加结构 token 与共享的 **2048 → 1** 标量评分头。从输入中的 `DECISION` 位置读取 hidden state，不执行词表 LM head，不生成自由文本。

候选名称和描述在运行时输入，同一个评分头可以处理不同任务定义。

### 候选上下文

- **A：独立候选。**每项评分只读取状态、问题和自身描述。
- **B：全候选可见。**每项评分还能读取完整候选集合，用于相对比较和动态 `other` 语义。

在小规模 pilot 中比较未见任务、动态 taxonomy、重叠类别和兜底选项，再选择 Choice 结构。Score 使用独立描述档位，Noul 使用 true／false 分支。

### 概率监督

混合经验证硬标签、实际事件结果、已知条件分布、适当降权的教师监督和经审核人类分布。主目标为按题交叉熵，通过消融决定是否加入 Brier 和一致性损失。

模型训练、温度拟合、工作流阈值选择和最终测试使用分离的数据。分布集中不等于正确率高，单题校准也不自动保证整个工作流可靠。

### 共享计算

先用普通 causal forward 建立参考实现，再研究 **状态 → 问题 → 候选** 的树形 attention，使用路径深度的 RoPE position，共享相同前缀计算。

先验证优化前后的 logits、loss 和梯度等价，再与包括前缀缓存在内的基线比较延迟和显存。

## 训练路线图

| 阶段 | 主要工作 | 初始数据预算 |
| --- | --- | --- |
| 定义任务 | 固定任务语义、数据划分、工作流与基线 | 代表性真实样本和规则样例 |
| 决策适配 | 训练评分头、结构 token 与问题理解能力 | 2 万–5 万条高质量决策题 |
| Pilot | 比较候选结构、初始化和监督来源 | 累计约 10 万条训练题 |
| 工程验证 | 验证共享计算等价性与性能 | Pilot 数据和边界样例 |
| 主训练 | 扩展任务覆盖、蒸馏能力 | 累计 50 万–200 万条训练题 |
| 难例改进 | 经验证难例与普通数据回放 | 10 万–30 万难例加回放 |
| 校准与验收 | 拟合温度、选业务阈值、测试冻结系统 | 独立校准、策略和测试数据 |

上表为计划预算。一个 decision 指一道带有完整候选集合的问题，预览版模型卡会记录实际训练数据和完成的阶段。

## 开始阅读

从[总体设计](docs/openjev_qwen3_design.md)开始，再阅读[模型架构](docs/openjev_model_architecture.md)和[训练流程](docs/openjev_training_pipeline.md)。详细研究文档目前为中文，欢迎贡献英文翻译。

模型下载地址和推理说明将随预览版发布，版本信息见[模型卡](docs/MODEL_CARD.md)。

本地文档检查仅依赖 **Python 3.10 或更新版本**：

```bash
python3 scripts/check_docs.py
```

## 文档导航

| 文档 | 内容 |
| --- | --- |
| [总体设计](docs/openjev_qwen3_design.md) | 研究取舍、能力边界和里程碑 |
| [模型架构](docs/openjev_model_architecture.md) | 输入编译、A／B 结构、attention、读出及推理 |
| [训练流程](docs/openjev_training_pipeline.md) | 九个执行步骤、八类数据配方、目标与退出条件 |
| [路线图](ROADMAP.md) | 模型发布承诺和长期工作 |
| [模型卡](docs/MODEL_CARD.md) | 模型状态、训练信息、评估和局限 |
| [发布指南](docs/RELEASING.md) | 可复现发布需要的产物与证据 |
| [变更记录](CHANGELOG.md) | 仓库变更和实际发布记录 |

## 参与贡献

欢迎架构讨论、数据生成器与验证器、参考实现、评估任务、文档修订和翻译。请先阅读[贡献指南](CONTRIBUTING.md)，Issue 和 PR 均可使用中文或英文。

Bug、功能提议和研究讨论请提交 Issue。社区协作遵循[行为准则](CODE_OF_CONDUCT.md)，敏感问题按照[安全说明](SECURITY.md)处理。

## 许可与引用

OpenJev 原创贡献采用 [Apache-2.0](LICENSE) 许可。第三方研究摘录和上游产物保留各自条款，详见 [NOTICE](NOTICE) 与[第三方说明](THIRD_PARTY_NOTICES.md)。每次模型发布都会注明具体产物的许可和上游要求。

引用项目可使用 [CITATION.cff](CITATION.cff)，模型发布后会补充带版本的模型引用信息。

## 致谢

感谢[计算机软件新技术国家重点实验室](https://cs.nju.edu.cn/cs_en/52964/list.htm)和[南京大学机器学习与数据挖掘研究所（LAMDA）](https://www.lamda.nju.edu.cn/)为 OpenJev 提供算力支持。

感谢 Qwen 团队提供基座模型，感谢 TypeSafe AI 分享 System One 模型方向。
