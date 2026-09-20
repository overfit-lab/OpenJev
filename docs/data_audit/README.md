# 当前数据状态

本目录只保留现用来源、最新制题审查和就绪状态。历史盘点、v1／v2 实验、旧工程训练报告与原始批次已归档，见[归档记录](../ARCHIVE.md)。

| 文档 | 内容 |
| --- | --- |
| [current_sources.json](current_sources.json) | 当前五份冻结划分、26 个来源的相对定位及文件哈希 |
| [teacher_codebook.md](teacher_codebook.md) | 本地 tokenizer 的 255 个双字母单 token 编码及范围限制 |
| [teacher_codebook.json](teacher_codebook.json) | 编码计数、tokenizer 哈希和验证范围 |
| [pipeline_canary.md](pipeline_canary.md) | 新制题与本地教师采集的实际小试、语义缺陷和覆盖缺口 |

## 当前冻结来源

| 划分 | 来源组 | 用户轮次前缀 |
| --- | ---: | ---: |
| train | 4,299 | 6,095 |
| dev_model | 160 | 222 |
| calibration | 160 | 218 |
| dev_policy | 120 | 171 |
| test | 120 | 126 |

来源目录为 outputs/chat_splits_v2_combined。名称中的 v2 是来源冻结版本，不是旧训练格式；新版制题仍依赖这份不可变的原文划分。不得因为文件名较旧而重划、改写或删除。原始来源池和合并前中间文件已归档，当前划分具备继续制题及回放所需的完整内容。

所有派生题继承来源组及 split；train 与保护集合不混用。已有分组包含精确重复与有限词面检查，不能证明完全排除语义改写和跨语言泄漏。上游许可、原始划分和真人作者身份仍未逐来源确认，不能将本地 chat 数据一概称为真人流量。

## 当前产物

- outputs/chat_splits_v2_combined：五份冻结聊天来源和防泄漏划分，继续用于制题。
- outputs/teacher_codebook_local_v1：255 个双字母单 token 编码、tokenizer 哈希及审计快照。

- outputs/questions_api_current、outputs/questions_review：当前无标签试制题与逐题复查。
- outputs/teacher_local_canary：本地教师工程小试，共 2 题、8 条轨迹，全部保持质量隔离。

旧硬标签 trial／review／train、远程教师预检及报告已整体归档，不再作为当前数据入口。归档位置、清单与恢复方法见[归档记录](../ARCHIVE.md)。当前没有可训练的 teacher_soft 数据集。

原文只存本地忽略目录；公开报告不包含原始聊天、本机绝对路径或教师凭据。后续操作见[执行说明](../execution.md)。
