# 本地教师的 255 候选编码审计

日期：2026-09-20。对象：Huihui-Qwen3.8-27B-abliterated-NVFP4-v4 的本地 tokenizer。

**结论：不受 A–Z 的 26 个字母限制。** 一个 token 可以包含多个字符。本次枚举 AA–ZZ 共 676 个双大写字母编码，其中 562 个在三个测试前缀后均为独立单 token，足以构造 255 个选项的编码表。已经按固定种子选出 255 个不同编码与 token ID，未给教师新增 token。

| 编码 | 实测 token ID | token 数 |
| --- | --- | ---: |
| A | 32 | 1 |
| AA | 5840 | 1 |
| AB | 1803 | 1 |
| AZ | 22828 | 1 |
| BA | 6844 | 1 |
| ZZ | 32424 | 1 |
| 000 | 15, 15, 15 | 3 |
| 254 | 17, 20, 19 | 3 |

这不是“所有两个字母都是单 token”：114 个组合未通过。数字 000–254 均拆成三个 token，不能取它们的首 token 概率作为整项概率。特殊 token 也不能随意新增后使用，因为教师没有学过它们的答案含义。

## 验证了什么

scripts/teacher_codes.py 使用 tokenizer.json 实际编码，而非只查词表。每个编码必须：

1. 单独编码只得到一个 token ID。
2. 在每个测试前缀后满足 encode(prefix+code)=encode(prefix)+[token_id]。
3. 该 token 解码回原字符串，编码与 token ID 均不重复。

三个测试前缀包括单独的 `Answer:` 加换行、思考结束标记后的答案前缀，以及带 assistant／thinking 模板的中文短示例。通过后从排序后的可用池用 seed=20260920 抽取 255 个编码，保留随机顺序与全部 ID。

tokenizer SHA-256：06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523。

## 在教师打分中如何使用

每题按固定种子建立可逆的“编码 → 原候选语义名称”映射，完整候选列表都提供给教师。教师 thinking 之后，以约定前缀进入答案位置，取**生成编码之前**的完整词表 logits。对于 K 个选项，直接 gather K 个已知 token ID，再对合法集合做 log_softmax，不需要让教师生成 K 次答案。

例如，编码 AA／AB／AC 可以分别映射三个语义选项；映射必须随记录保存，不能假设 AA 永远对应首选答案。单条样本的 logits 不按 top-k 裁剪，所有合法候选都应存在。学生使用原 criteria 名称与描述，编码表不成为学生的 255 类分类头。

后续采集仍需验证实际完整 thinking 与答案前缀上的编码边界，并记录合法候选的完整词表概率质量。检查编码和选项置换后，按语义名称还原并比较概率，避免把编码偏好当成任务知识。

## 产物与限制

本地 outputs/teacher_codebook_local_v1 包含 codebook.json、代码快照和 manifest；目录受 Git 忽略。机器可读的公开摘要见[编码摘要](teacher_codebook.json)。工具只通过运行时参数读取 tokenizer，产物没有机器绝对路径。命令见[执行说明](../execution.md)。

这次 tokenizer 审计没有加载模型或验证 255 选项决策能力。后续已在 A800 上完成 2 题、8 条 thinking／logits 轨迹，见[流水线小试](pipeline_canary.md)；该结果不构成 255 项能力或训练集验收。

这份表绑定上述 tokenizer。换 tokenizer 或聊天模板后必须重新检查；如果可用单 token 不足，再设计固定宽度多 token 编码的完整序列打分。主线与 K 数量分布见[三阶段训练计划](../openjev_training_pipeline.md)。
