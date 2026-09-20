<p align="center">
  <img src="docs/assets/openjev-banner.svg" alt="OpenJev — 面向软件的类型化决策模型" width="100%" />
</p>

<p align="center">
  <strong>面向软件的开源决策模型。</strong><br />
  直接输出可供代码使用的决策与概率，服务自动化工作流。
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-74e0c1" alt="许可证：Apache-2.0" /></a>
  <a href="ROADMAP.md"><img src="https://img.shields.io/badge/status-data%20preparation-7aa2f7" alt="当前阶段：数据准备" /></a>
  <a href="ROADMAP.md"><img src="https://img.shields.io/badge/model%20preview-coming%20soon-f5c97a" alt="模型预览版即将发布" /></a>
</p>

<p align="center">
  <a href="README.md">English</a> · 简体中文 · <a href="docs/README.md">文档</a> · <a href="ROADMAP.md">路线图</a> · <a href="CONTRIBUTING.md">参与贡献</a>
</p>

> **首个模型预览版即将发布。**
> 预览版将提供模型权重、推理示例和初步评估，进度见[路线图](ROADMAP.md)。

目前仓库包含架构设计、训练方案、[聊天数据流程](docs/openjev_training_pipeline.md)，以及本地语料准备、离线数据检查和评分模型工程参考。公共输入编译器、三类 CLI／HTTP 响应、PiSSA 模型与检查点组件及教师编码审计已实现；旧硬标签流程已归档；API 制题、语义审核、离线回放及本地教师采集入口已实现，训练集验收与软训练器仍待完成。正式训练模型尚未发布。

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

三个 primitive 复用一个评分模型；接口已实现，模型能力仍待训练验收：

| 类型 | 问题形式 | 预期输出 |
| --- | --- | --- |
| **Choice** | 哪个候选最符合当前输入？ | 选中的候选名称和完整候选概率 |
| **Score** | 当前输入落在哪些描述性档位？ | 档位概率、加权分数、legend 和 confidence |
| **Noul** | 这个命题是否成立？ | 回答为真的概率 |

请求采用 Jev 的 `state + questions` 结构，候选放在 `criteria` 中。

```json
{
  "model": "openjev-preview",
  "state": "Tracking says delivered, but I have not received my parcel.",
  "questions": {
    "route": {
      "type": "choice",
      "instructions": "Which team should handle this first?",
      "criteria": {
        "shipping": "Delivery and missing parcels",
        "billing": "Charges and invoices",
        "returns": "Returns of received items"
      }
    }
  }
}
```

响应结构如下，概率为演示值；本地接口已实现，正式训练模型尚未发布：

```json
{
  "model": "openjev-preview",
  "answers": {
    "route": {
      "type": "choice",
      "choice": "shipping",
      "confidence": 0.83,
      "probabilities": {"shipping": 0.90, "billing": 0.03, "returns": 0.07}
    }
  },
  "usage": {"input_tokens": 210, "output_tokens": 31}
}
```

`confidence` 采用 OpenJev 定义的前两项概率差，不表示正确率，也不宣称与 Jev 官方数值等价。完整字段与兼容边界见[接口约定](docs/openjev_api_contract.md)。

## 技术路线

```mermaid
flowchart LR
    A[状态与类型化问题] --> B[输入编译器]
    B --> C[可配置的 causal backbone]
    C --> D[共享标量评分头]
    D --> E[按问题归一化的概率]
    E --> F[类型化输出]
    F --> G[工作流决策策略]
```

### 候选评分

基座型号可配置，保留所选模型的 embedding 和 Transformer 层，增加结构 token 与共享的 **hidden_size → 1** 标量评分头。每次实验固定 checkpoint 和 tokenizer。从输入中的 `DECISION` 位置读取 hidden state，不执行词表 LM head，不生成自由文本。

候选名称和描述在运行时输入，同一个评分头可以处理不同任务定义。

### 候选上下文

- **A：独立候选。**每项评分只读取状态、问题和自身描述。
- **B：全候选可见。**每项评分还能读取完整候选集合，用于相对比较和动态 `other` 语义。

首轮采用 B；A 留作后续对照，考察未见任务、动态 taxonomy、重叠类别和兜底选项。Score 使用独立描述档位，Noul 使用 true／false 分支。

### 概率监督

API 从已有对话构造问题和选项，解析 JSON 并检查质量；本地教师先推理，再提供完整候选概率，学生直接进行 PiSSA 软蒸馏。教师思维链和临时编码不进入学生输入。新的制题和本地概率采集已有执行入口，保存原始响应、思考轨迹和 logits 供审计。[小试报告](docs/data_audit/pipeline_canary.md)记录实际覆盖与拒收样例；目前还没有通过训练验收的软标签数据集。

模型训练、温度拟合、工作流阈值选择和最终测试使用分离的数据。分布集中不等于正确率高，单题校准也不自动保证整个工作流可靠。

### 共享计算

先用普通 causal forward 建立参考实现，再研究 **状态 → 问题 → 候选** 的树形 attention，使用路径深度的 RoPE position，共享相同前缀计算。

先验证优化前后的 logits、loss 和梯度等价，再与包括前缀缓存在内的基线比较延迟和显存。

## 训练路线图

| 阶段 | 工作 | 首版安排 |
| --- | --- | --- |
| 1. 准备数据 | 真实聊天 → API 制题 → JSON／语义检查 → 本地教师 thinking 与完整概率 → 冻结 | 先验收约 100 条，再扩至 1K–3K，覆盖 2–255 个候选 |
| 2. PiSSA 直接软蒸馏 | 从选定基座联合训练低秩参数、评分头与结构 token，只用教师分布监督 | 不做硬标签预热 |
| 3. 验证 | 开发评估、独立校准、策略选择和最终测试 | 报告质量、成本与支持范围 |

首版执行 **1 → 2 → 3**，RLCD 留作后续研究。标准 PiSSA 冻结残差基底，通过低秩项更新有效主干权重；不单独预热评分头，也不先训练 LoRA。具体参数范围与实现缺口见[训练计划](docs/openjev_training_pipeline.md)。

## 开始阅读

从[总体设计](docs/openjev_design.md)开始，再阅读[模型架构](docs/openjev_model_architecture.md)和[训练流程](docs/openjev_training_pipeline.md)。详细研究文档目前为中文，欢迎贡献英文翻译。

本地推理命令见[执行说明](docs/execution.md)，模型下载地址将随预览版发布，版本信息见[模型卡](docs/MODEL_CARD.md)。

本地文档检查仅依赖 **Python 3.10 或更新版本**：

```bash
python3 scripts/check_docs.py
```

## 文档导航

| 文档 | 内容 |
| --- | --- |
| [总体设计](docs/openjev_design.md) | 研究取舍、能力边界和里程碑 |
| [模型架构](docs/openjev_model_architecture.md) | 输入编译、A／B 结构、attention、读出及推理 |
| [训练流程](docs/openjev_training_pipeline.md) | 三个阶段、API 制题、本地教师概率与 PiSSA 软蒸馏 |
| [训练数据格式](docs/openjev_training_data.md) | 无标签题库、教师分布、来源记录与回放 |
| [接口约定](docs/openjev_api_contract.md) | Jev 输入输出字段、示例与兼容边界 |
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
